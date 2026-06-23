import torch as torch
from torchvision import transforms
from tqdm.notebook import tqdm
import numpy as np
import cv2
import os
from PIL import Image
from collections import OrderedDict
import shutil
from typing import Dict, Optional, Any

class CacheManager:
    """
    Dynamic Cache Manager with LRU (Least Recently Used) Eviction.

    Manages a bounded cache directory, automatically evicting oldest-accessed
    files when the cache exceeds the configured size limit.

    Features:
    - Auto-detection: Automatically detects available storage and reserves space
    - Warm caching: Pre-caches videos up to the limit before training starts
    - On-demand caching: Videos are cached only when first accessed
    - LRU eviction: Oldest-accessed files deleted first when limit exceeded
    - O(1) operations: All cache operations are constant time
    """

    def __init__(
        self,
        cache_dir: str,
        max_size_gb: Optional[float] = None,
        transforms: Optional[Any] = None,
        num_frames: int = 30,
        eviction_check_interval: int = 10,
        eviction_buffer_percent: float = 0.10,
        auto_detect: bool = True,
        reserve_gb: float = 10.0
    ) -> None:
        """
        Initialize the CacheManager.

        Args:
            cache_dir: Directory to store cached .npy files
            max_size_gb: Maximum cache size in GB (None = auto-detect)
            transforms: Torchvision transforms to apply when caching
            num_frames: Number of frames per video
            eviction_check_interval: Check for eviction every N cache misses
            eviction_buffer_percent: Extra space to free during eviction (0.10 = 10%)
            auto_detect: If True, auto-detect available storage
            reserve_gb: GB to keep free when auto-detecting
        """
        self.cache_dir = cache_dir
        self.transforms = transforms
        self.num_frames = num_frames
        self.eviction_check_interval = eviction_check_interval
        self.eviction_buffer_percent = eviction_buffer_percent
        self.reserve_bytes = reserve_gb * (1024 ** 3)

        os.makedirs(cache_dir, exist_ok=True)

        # Auto-detect or use provided max_size_gb
        if auto_detect and max_size_gb is None:
            self.max_size_bytes = self._calculate_max_cache_size()
            self.auto_detected = True
        else:
            self.max_size_bytes = (max_size_gb or 7.5) * (1024 ** 3)
            self.auto_detected = False

        # Incremental size tracking - O(1)
        self.current_size_bytes: int = 0

        # LRU tracking with OrderedDict - O(1) for all operations
        self.lru_cache: OrderedDict[str, int] = OrderedDict()

        # Statistics
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.warm_cached: int = 0
        self.misses_since_eviction_check: int = 0

        # Initialize from existing files
        self._initialize_from_disk()

        detect_str = "(auto-detected)" if self.auto_detected else "(configured)"
        print(f"✓ CacheManager initialized {detect_str}")
        print(f"  - Max size: {self.max_size_bytes / (1024**3):.1f} GB")
        print(f"  - Current: {self.get_cache_size_gb():.2f} GB ({len(self.lru_cache)} files)")

    def _calculate_max_cache_size(self) -> int:
        """Calculate maximum cache size based on available disk space."""
        try:
            disk_usage = shutil.disk_usage(self.cache_dir)
            max_cache_bytes = max(0, disk_usage.free - self.reserve_bytes)

            print(f"✓ Storage auto-detection:")
            print(f"  - Free: {disk_usage.free / (1024**3):.1f} GB")
            print(f"  - Reserved: {self.reserve_bytes / (1024**3):.1f} GB")
            print(f"  - Available for cache: {max_cache_bytes / (1024**3):.1f} GB")

            return max_cache_bytes
        except Exception as e:
            print(f"⚠ Auto-detect failed: {e}. Using 7.5 GB default.")
            return int(7.5 * (1024 ** 3))

    def _initialize_from_disk(self) -> None:
        """Load existing cache files and calculate initial size."""
        if not os.path.exists(self.cache_dir):
            return

        cached_files = []
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.npy'):
                filepath = os.path.join(self.cache_dir, filename)
                file_size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                cached_files.append((filename, file_size, mtime))
                self.current_size_bytes += file_size

        # Sort by modification time (oldest first)
        cached_files.sort(key=lambda x: x[2])
        for filename, file_size, _ in cached_files:
            self.lru_cache[filename] = file_size

    def get_cache_size_gb(self) -> float:
        """Get current cache size in GB."""
        return self.current_size_bytes / (1024 ** 3)

    def get_or_create(self, video_name: str, video_path: str) -> torch.Tensor:
        """
        Get cached tensor or create it from video.

        Args:
            video_name: Name of the video file (e.g., 'video_001.mp4')
            video_path: Full path to the video file

        Returns:
            Video frames tensor [num_frames, 3, H, W]
        """
        cache_filename = video_name.replace('.mp4', '.npy')
        cache_path = os.path.join(self.cache_dir, cache_filename)

        # CACHE HIT
        if cache_filename in self.lru_cache:
            self.hits += 1
            self.lru_cache.move_to_end(cache_filename)  # O(1)
            return torch.from_numpy(np.load(cache_path))

        # CACHE MISS
        self.misses += 1
        self.misses_since_eviction_check += 1
        video_tensor = self._decode_video(video_path)

        # Save to cache
        np.save(cache_path, video_tensor.numpy())
        file_size = os.path.getsize(cache_path)
        self.lru_cache[cache_filename] = file_size
        self.current_size_bytes += file_size

        # Lazy eviction
        if self.misses_since_eviction_check >= self.eviction_check_interval:
            self._batch_evict_if_needed()
            self.misses_since_eviction_check = 0

        return video_tensor

    def _decode_video(self, video_path: str) -> torch.Tensor:
        """Decode video file to tensor."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                frame_tensor = torch.zeros((3, HEIGHT, WIDTH))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transforms:
                    frame = Image.fromarray(frame)
                    frame_tensor = self.transforms(frame)
                else:
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame_tensor)

        cap.release()
        return torch.stack(frames, dim=0)

    def _batch_evict_if_needed(self) -> None:
        """Batch eviction with buffer."""
        if self.current_size_bytes <= self.max_size_bytes:
            return

        excess_bytes = self.current_size_bytes - self.max_size_bytes
        buffer_bytes = self.max_size_bytes * self.eviction_buffer_percent
        bytes_to_evict = excess_bytes + buffer_bytes

        bytes_evicted = 0
        files_to_remove = []

        for filename, file_size in self.lru_cache.items():
            if bytes_evicted >= bytes_to_evict:
                break
            files_to_remove.append((filename, file_size))
            bytes_evicted += file_size

        for filename, file_size in files_to_remove:
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
            del self.lru_cache[filename]
            self.current_size_bytes -= file_size
            self.evictions += 1

    def warm_cache(self, video_folder: str, max_videos: Optional[int] = None) -> int:
        """Pre-cache videos up to storage limit before training."""
        print(f"\n{'='*40}\nWARM CACHE\n{'='*40}")

        if self.auto_detected:
            self._calculate_max_cache_size()

        video_list = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        if max_videos:
            video_list = video_list[:max_videos]

        uncached = [v for v in video_list if v.replace('.mp4', '.npy') not in self.lru_cache]
        print(f"To cache: {len(uncached)} videos")

        avg_size = sum(self.lru_cache.values()) / len(self.lru_cache) if self.lru_cache else SEQ_LEN * 3 * HEIGHT * WIDTH * 4
        cached_count = 0

        for video_name in tqdm(uncached, desc="Caching"):
            if (self.max_size_bytes - self.current_size_bytes) < avg_size * 1.1:
                print(f"⚠ Storage limit reached. Cached {cached_count} videos.")
                break

            video_path = os.path.join(video_folder, video_name)
            cache_filename = video_name.replace('.mp4', '.npy')
            cache_path = os.path.join(self.cache_dir, cache_filename)

            try:
                video_tensor = self._decode_video(video_path)
                np.save(cache_path, video_tensor.numpy())
                file_size = os.path.getsize(cache_path)
                self.lru_cache[cache_filename] = file_size
                self.current_size_bytes += file_size
                cached_count += 1
                self.warm_cached += 1
                avg_size = (avg_size * 0.9) + (file_size * 0.1)
                del video_tensor
            except Exception as e:
                print(f"⚠ Failed: {video_name}: {e}")

        print(f"✓ Cached {cached_count} videos ({self.get_cache_size_gb():.2f} GB)")
        return cached_count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': (self.hits / total * 100) if total > 0 else 0,
            'evictions': self.evictions,
            'cache_size_gb': self.get_cache_size_gb(),
            'num_files': len(self.lru_cache)
        }

    def reset_epoch_stats(self) -> None:
        """Reset hit/miss/eviction counters for new epoch."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.misses_since_eviction_check = 0

    def print_stats(self, epoch: Optional[int] = None) -> None:
        """Print formatted cache statistics."""
        stats = self.get_stats()
        prefix = f"Epoch {epoch} " if epoch else ""
        print(f"{prefix}Cache: Hits={stats['hits']} Miss={stats['misses']} "
              f"Rate={stats['hit_rate']:.0f}% Evict={stats['evictions']}")


# Initialize CacheManager with config
cache_transforms = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor()
])

cache_manager = CacheManager(
    cache_dir=CACHE_DIR,
    max_size_gb=None,
    transforms=cache_transforms,
    num_frames=SEQ_LEN,
    eviction_check_interval=CONFIG.eviction_check_interval,
    eviction_buffer_percent=CONFIG.eviction_buffer_percent,
    auto_detect=True,
    reserve_gb=CONFIG.reserve_gb
)

# Optional: Pre-cache videos
cache_manager.warm_cache(VIDEO_DIR)