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
from torch.nn.parallel import DistributedDataParallel as DDP 
from mmdet.apis import init_detector, inference_detector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, config_path, checkpoint_path, video_path, output_dir, rank=0, world_size=1):
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        self.frames_extracted = False
        logger.info(f"Initialized VideoProcessor: rank={rank}, world_size={world_size}, device={self.device}, video={video_path}")

    def setup(self):
        logger.info("Entering setup method")
        if self.rank == 0:
            logger.info("Setting up environment...")
        
        # Initialize distributed process group for multi-GPU
        if self.world_size > 1:
            logger.info(f"Initializing distributed process group for world_size={self.world_size}")
            try:
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.rank
                )
                logger.info("Distributed process group initialized")
            except Exception as e:
                logger.error(f"Failed to initialize process group: {str(e)}")
                raise
        
        # Create output directory
        logger.info(f"Creating output directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory
        logger.info("Creating temporary directory")
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Temp directory created: {self.temp_dir}")
        
        # Initialize model
        logger.info("Initializing model...")
        try:
            self.model = init_detector(
                str(self.config_path),
                str(self.checkpoint_path),
                device=self.device
            )
            if self.world_size > 1:
                logger.info(f"Wrapping model with DDP on rank {self.rank}")
                self.model = DDP(self.model, device_ids=[self.rank])
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed on rank {self.rank}: {str(e)}")
            raise
        logger.info("Exiting setup method")

    def extract_frames(self):
        if self.rank != 0:
            logger.info("Skipping extract_frames: not rank 0")
            return []
        
        logger.info(f"Starting frame extraction from {self.video_path} to {self.temp_dir}")
        if not self.video_path.exists():
            logger.error(f"Video file does not exist: {self.video_path}")
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")
        
        output_pattern = self.temp_dir / "frame_%06d.jpg"
        command = [
            "ffmpeg",
            "-i", str(self.video_path),
            "-vf", "fps=15,scale=1280:720",
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
            logger.error(f"Unexpected error during frame extraction: {str(e)}")
            raise
        
        frame_paths = sorted(self.temp_dir.glob("frame_*.jpg"))
        logger.info(f"Extracted {len(frame_paths)} frames")
        if frame_paths:
            self.frames_extracted = True
            logger.info("Frames extracted successfully: frames_extracted=True")
        else:
            logger.error("No frames extracted")
        return frame_paths

    def overlay_bboxes(self, image_path: Path, result: list, output_path: Path, score_thr: float = 0.1):
        logger.info(f"Overlaying bboxes on {image_path}")
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        bboxes = result[0]  # Single class
        for bbox in bboxes:
            if bbox[4] >= score_thr:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{bbox[4]:.2f}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(output_path), img)
        logger.info(f"Saved overlaid frame: {output_path}")

    def run_inference(self, frame_paths: list):
        logger.info(f"Running inference on {len(frame_paths)} frames")
        results_dir = self.output_dir / "results"
        overlay_dir = self.output_dir / "overlays"
        results_dir.mkdir(exist_ok=True)
        overlay_dir.mkdir(exist_ok=True)
        
        for frame_path in frame_paths:
            try:
                result = inference_detector(self.model, str(frame_path))
                logger.info(f"Detections for {frame_path.name}: {len(result[0] if isinstance(result, list) else result)} bboxes")
                output_path = results_dir / frame_path.name
                self.model.show_result(
                    str(frame_path),
                    result,
                    score_thr=0.1,
                    out_file=str(output_path)
                )
                overlay_path = overlay_dir / frame_path.name
                self.overlay_bboxes(frame_path, result, overlay_path)
            except Exception as e:
                logger.warning(f"Failed to process frame {frame_path.name}: {str(e)}")
                continue

    def create_output_video(self):
        if self.rank != 0:
            logger.info("Skipping create_output_video: not rank 0")
            return
        
        logger.info("Creating output video from overlaid frames...")
        overlay_dir = self.output_dir / "overlays"
        output_video = self.output_dir / "output_video.mp4"
        first_frame = next(overlay_dir.glob("*.jpg"), None)
        if not first_frame:
            logger.error("No overlaid frames found")
            return
        
        img = cv2.imread(str(first_frame))
        if img is None:
            logger.error(f"Failed to load first frame: {first_frame}")
            return
        h, w = img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_video), fourcc, 15.0, (w, h))
        
        for frame_path in sorted(overlay_dir.glob("*.jpg")):
            img = cv2.imread(str(frame_path))
            if img is not None:
                writer.write(img)
        writer.release()
        logger.info(f"Output video saved to {output_video}")

    def cleanup(self):
        if self.rank == 0 and hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Destroyed distributed process group")

    def process(self):
        logger.info("Entering process method")
        try:
            logger.info(f"Starting process on rank {self.rank}")
            self.setup()
            logger.info("Setup completed, starting frame extraction")
            frame_paths = self.extract_frames()
            logger.info(f"Frame extraction returned {len(frame_paths)} frames")
            logger.info(f"Frames extracted successfully: {self.frames_extracted}")
            self.run_inference(frame_paths)
            self.create_output_video()
            if self.rank == 0:
                logger.info(f"Processing complete. Results saved to {self.output_dir}")
        except Exception as e:
            logger.error(f"Processing failed on rank {self.rank}: {str(e)}")
            raise
        finally:
            self.cleanup()
        logger.info("Exiting process method")

def main():
    logger.info("Entering main function")
    parser = argparse.ArgumentParser(description="Process video for object detection")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='output')
    args = parser.parse_args()
    
    logger.info(f"Arguments: config={args.config}, checkpoint={args.checkpoint}, video={args.video}, output_dir={args.output_dir}")
    
    # For single-GPU, use rank=0, world_size=1
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    processor = VideoProcessor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        video_path=args.video,
        output_dir=args.output_dir,
        rank=rank,
        world_size=world_size
    )
    processor.process()
    logger.info("Exiting main function")

if __name__ == "__main__":
    main()