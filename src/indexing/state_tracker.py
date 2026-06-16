import os
import json
import logging
from pathlib import Path
import config

logger = logging.getLogger(__name__)

class DocumentStateTracker:
    """
    Scans the document directory, compares the current PDF states with a cached snapshot,
    and determines if the document index needs to be rebuilt.
    """

    def __init__(self, data_dir: Path = None, cache_file: Path = None):
        """
        Args:
            data_dir: Path to directory containing raw documents.
            cache_file: Path to JSON cache file storing state.
        """
        self.data_dir = data_dir or config.DATA_DIR
        self.cache_file = cache_file or config.INDEX_STATE_FILE

    def get_current_state(self) -> dict:
        """
        Scans data_dir for all PDFs and maps filenames to their metadata.
        
        Returns:
            Dictionary containing:
                {
                    "files": {
                        "filename.pdf": {
                            "size": int (file size in bytes),
                            "modified": int (last modified timestamp as int)
                        }
                    }
                }
        """
        current_files = {}
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return {"files": current_files}

        for file_path in self.data_dir.glob("**/*.pdf"):
            try:
                stat = file_path.stat()
                current_files[file_path.name] = {
                    "size": stat.st_size,
                    "modified": int(stat.st_mtime)
                }
            except Exception as e:
                logger.error(f"Error accessing stats for file {file_path.name}: {e}")

        return {"files": current_files}

    def load_cached_state(self) -> dict:
        """
        Loads the cached document state from cache_file.
        
        Returns:
            Dictionary representing the cached index state.
        """
        if not self.cache_file.exists():
            logger.info("No cached index state file found.")
            return {"files": {}}

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                # Ensure the root dictionary contains the expected "files" key
                if "files" not in state:
                    state = {"files": {}}
                logger.info("Loaded cached document state snapshot successfully.")
                return state
        except Exception as e:
            logger.warning(f"Failed to read cached index state file: {e}. Defaulting to empty state.")
            return {"files": {}}

    def save_state(self, state: dict):
        """
        Saves the current document state snapshot to cache_file.
        
        Args:
            state: The state snapshot dictionary to cache.
        """
        try:
            os.makedirs(self.cache_file.parent, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
            logger.info(f"Saved new document state snapshot to: {self.cache_file}")
        except Exception as e:
            logger.error(f"Error persisting index state file: {e}")

    def is_index_stale(self, current_state: dict, cached_state: dict) -> bool:
        """
        Compares the current file state snapshot with the cached snapshot.
        The index is considered stale if any file is added, removed, or modified.
        
        Args:
            current_state: Current scanned state dict.
            cached_state: Loaded cached state dict.
            
        Returns:
            True if the index is stale and a rebuild is required, False otherwise.
        """
        current_files = current_state.get("files", {})
        cached_files = cached_state.get("files", {})

        if not cached_files:
            logger.info("Index is stale: No cached index state exists.")
            return True

        # 1. Compare file lists
        current_filenames = set(current_files.keys())
        cached_filenames = set(cached_files.keys())

        if current_filenames != cached_filenames:
            added = current_filenames - cached_filenames
            removed = cached_filenames - current_filenames
            if added:
                logger.info(f"Index is stale: Added files detected: {added}")
            if removed:
                logger.info(f"Index is stale: Removed files detected: {removed}")
            return True

        # 2. Check metadata details for differences
        for filename, current_info in current_files.items():
            cached_info = cached_files.get(filename)
            if not cached_info:
                logger.info(f"Index is stale: Missing state for {filename}")
                return True

            if current_info["size"] != cached_info["size"]:
                logger.info(
                    f"Index is stale: File size changed for {filename} "
                    f"(current: {current_info['size']} bytes, cached: {cached_info['size']} bytes)"
                )
                return True

            if current_info["modified"] != cached_info["modified"]:
                logger.info(
                    f"Index is stale: Modification time changed for {filename} "
                    f"(current: {current_info['modified']}, cached: {cached_info['modified']})"
                )
                return True

        logger.info("Index is up to date: No modifications detected in the PDF files.")
        return False
