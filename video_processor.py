import os
import subprocess
import logging
import argparse
from pathlib import Path
import tempfile
import shutil
import cv2
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import get_device, setup_multi_processes
from mmcv import Config
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Class to handle video frame extraction, multi-GPU inference, and output video creation."""
    
    def __init__(self, config_path: str, checkpoint_path: str, video_path: str, output_dir: str, rank: int, world_size: int):
        """
        Initialize the video processor.
        
        Args:
            config_path (str): Path to the model configuration file
            checkpoint_path (str): Path to the model checkpoint file
            video_path (str): Path to the input video
            output_dir (str): Directory to save output frames and results
            rank (int): Rank of the current process
            world_size (int): Total number of processes (GPUs)
        """
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.world_size = world_size
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.temp_dir = None
        
    def setup(self):
        """Set up the environment, model, and output directories."""
        if self.rank == 0:
            logger.info("Setting up environment...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory for frames
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Initialize model
        try:
            self.model = init_detector(
                str(self.config_path),
                str(self.checkpoint_path),
                device=self.device
            )
            self.model = DDP(self.model, device_ids=[self.rank])
            if self.rank == 0:
                logger.info("Model initialized successfully on all GPUs")
        except Exception as e:
            logger.error(f"Failed to initialize model on rank {self.rank}: {e}")
            raise
    
    def extract_frames(self):
        """Extract frames from video using ffmpeg (only on rank 0)."""
        if self.rank != 0:
            return []
        
        logger.info(f"Starting frame extraction from {self.video_path} to {self.temp_dir}")
        if not self.video_path.exists():
            logger.error(f"Video file does not exist: {self.video_path}")
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")
        
        output_pattern = self.temp_dir / "frame_%06d.jpg"
        command = [
            "ffmpeg",
            "-i", str(self.video_path),
            "-vf", "fps=15,scale=1280:720",  # Lower fps and resolution
            "-q:v", "2",
            str(output_pattern)
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(command)}")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"Frame extraction completed: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during frame extraction: {e}")
            raise
        
        frame_paths = sorted(self.temp_dir.glob("frame_*.jpg"))
        logger.info(f"Extracted {len(frame_paths)} frames")
        if not frame_paths:
            logger.error("No frames extracted")
        return frame_paths
    
    def overlay_bboxes(self, image_path: Path, result: list, output_path: Path, score_thr: float = 0.3):
        """Overlay bounding boxes on the image and save to output path."""
        img = cv2.imread(str(image_path))
        bboxes = result[0]  # Assuming single class (uav_people)
        
        for bbox in bboxes:
            if bbox[4] >= score_thr:
                x1, y1, x2, y2 = map(int, bbox[:4])
                score = bbox[4]
                # Draw rectangle and score
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        cv2.imwrite(str(output_path), img)
    
    def run_inference(self, frame_paths: list):
        """Run inference on extracted frames using multiple GPUs."""
        logger.info(f"Running inference on rank {self.rank}...")
        
        results_dir = self.output_dir / "results"
        overlay_dir = self.output_dir / "overlays"
        if self.rank == 0:
            results_dir.mkdir(exist_ok=True)
            overlay_dir.mkdir(exist_ok=True)
        
        # Distribute frames across GPUs
        frames_per_gpu = len(frame_paths) // self.world_size
        start_idx = self.rank * frames_per_gpu
        end_idx = (self.rank + 1) * frames_per_gpu if self.rank < self.world_size - 1 else len(frame_paths)
        local_frames = frame_paths[start_idx:end_idx]
        
        for frame_path in local_frames:
            try:
                # Run inference
                result = inference_detector(self.model, str(frame_path))
                
                # Save original detection result
                output_path = results_dir / frame_path.name
                self.model.module.show_result(
                    str(frame_path),
                    result,
                    score_thr=0.3,
                    out_file=str(output_path)
                )
                
                # Overlay bounding boxes
                overlay_path = overlay_dir / frame_path.name
                self.overlay_bboxes(frame_path, result, overlay_path)
                
                logger.info(f"Rank {self.rank} processed frame: {frame_path.name}")
            except Exception as e:
                logger.warning(f"Rank {self.rank} failed to process frame {frame_path.name}: {e}")
                continue
    
    def create_output_video(self):
        """Create a video from overlaid frames (only on rank 0)."""
        if self.rank != 0:
            return
        
        logger.info("Creating output video from overlaid frames...")
        
        overlay_dir = self.output_dir / "overlays"
        output_video = self.output_dir / "output_video.mp4"
        
        # Get frame dimensions
        first_frame = next(overlay_dir.glob("*.jpg"), None)
        if not first_frame:
            logger.error("No overlaid frames found")
            return
        
        img = cv2.imread(str(first_frame))
        height, width = img.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            30.0,
            (width, height)
        )
        
        # Write frames to video
        for frame_path in sorted(overlay_dir.glob("*.jpg")):
            img = cv2.imread(str(frame_path))
            video_writer.write(img)
        
        video_writer.release()
        logger.info(f"Output video saved to {output_video}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                if self.rank == 0:
                    logger.info("Cleaned up temporary directory")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")
    
    def process(self):
        """Main processing pipeline."""
        try:
            self.setup()
            logger.info("Setup completed, starting frame extraction")
            frame_paths = self.extract_frames()
            logger.info(f"Frame extraction returned {len(frame_paths)} frames, or failed to execute")
            dist.barrier()  # Synchronize all processes
            
            # Broadcast frame paths to all ranks
            frame_paths = torch.tensor([len(frame_paths)], dtype=torch.int).to(self.device)
            dist.broadcast(frame_paths, src=0)
            frame_paths = [self.temp_dir / f"frame_{i:06d}.jpg" for i in range(1, frame_paths.item() + 1)] if self.rank != 0 else frame_paths
            
            self.run_inference(frame_paths)
            dist.barrier()  # Synchronize before video creation
            
            self.create_output_video()
            if self.rank == 0:
                logger.info(f"Processing complete. Results saved to {self.output_dir}")
        except Exception as e:
            logger.error(f"Processing failed on rank {self.rank}: {e}")
            raise
        finally:
            self.cleanup()

def setup_distributed(rank: int, world_size: int):
    """Setup distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def worker(rank: int, world_size: int, args):
    """Worker function for each GPU process."""
    setup_distributed(rank, world_size)
    # Load config
    cfg = Config.fromfile(args.config)
    setup_multi_processes(cfg)  # Pass Config object instead of dict
    
    processor = VideoProcessor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        video_path=args.video,
        output_dir=args.output_dir,
        rank=rank,
        world_size=world_size
    )
    
    processor.process()
    dist.destroy_process_group()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process video for object detection with multi-GPU support")
    parser.add_argument(
        "--config",
        default="configs/uav_people/co_dino_5scale_r50_1x_coco.py",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save output frames, overlays, and video"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    if world_size == 0:
        logger.error("No GPUs available")
        return
    
    logger.info(f"Using {world_size} GPUs")
    
    # Spawn processes
    mp.spawn(
        worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
