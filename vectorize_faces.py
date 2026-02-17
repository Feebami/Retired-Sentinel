import os
import sys
import pickle
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from itertools import combinations

import cv2
import torch
import numpy as np
from torchvision.utils import save_image
from facenet_pytorch import InceptionResnetV1, MTCNN

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

@dataclass
class Config:
    # Paths
    FACES_DIR = 'faces'
    OUTPUT_FILE = 'face_vectors.pkl'
    TEST_IMAGE_DIR = 'test_images'
    
    # Model Parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Core Classes
# -------------------------------------------------------------------------

class FaceEncoder:
    """Wrapper for Face Detection and Embedding Generation models."""
    def __init__(self, device: str):
        logger.info(f"Loading models on {device}...")
        self.device = device
        self.mtcnn = MTCNN(margin=40, device=device, thresholds=[0.5, 0.6, 0.6])
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def get_embedding(self, image_bgr: np.ndarray, save_face_path: str = None) -> Optional[np.ndarray]:
        """
        Takes a BGR image (OpenCV format), detects a face, and returns its embedding.
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Detect face
            face_tensor = self.mtcnn(image_rgb)
            if face_tensor is None:
                return None
                        
            # Save face tensor for debugging
            if save_face_path:
                save_image(face_tensor, save_face_path, normalize=True)

            # Generate embedding
            with torch.no_grad():
                # Move tensor to device if necessary
                face_tensor = face_tensor.to(self.device)
                embedding = self.resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None


class VectorManager:
    """Handles file I/O for images and vector storage."""
    def __init__(self, encoder: FaceEncoder):
        self.encoder = encoder
        self.vectors: Dict[str, List[np.ndarray]] = {}

    def process_directory(self, root_dir: str):
        """Iterates through subdirectories and generates vectors for all images."""
        if not os.path.exists(root_dir):
            logger.error(f"Directory not found: {root_dir}")
            return

        logger.info(f"Scanning directory: {root_dir}")
        
        for person_name in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            person_vectors = []
            image_files = os.listdir(person_dir)
            
            for i, img_name in enumerate(image_files):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                
                if img is None:
                    logger.warning(f"Skipping unreadable image: {img_path}")
                    continue

                embedding = self.encoder.get_embedding(img)
                if embedding is not None:
                    person_vectors.append(embedding)
                else:
                    logger.warning(f"No face detected in: {img_path}")

            if person_vectors:
                self.vectors[person_name] = person_vectors
                logger.info(f"Processed '{person_name}': {len(person_vectors)} vectors.")
            else:
                logger.warning(f"No valid vectors found for '{person_name}'.")

    def save_vectors(self, filepath: str):
        """Saves the dictionary of vectors to a pickle file."""
        if not self.vectors:
            logger.warning("No vectors to save.")
            return
            
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.vectors, f)
            logger.info(f"Successfully saved {len(self.vectors)} identities to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save vectors: {e}")

    def get_vectors(self):
        return self.vectors


class StatsAnalyzer:
    """Calculates and prints statistics about the vectors."""
    
    @staticmethod
    def print_intra_class_distances(vector_dict: Dict[str, List[np.ndarray]]):
        """Calculates consistency (average distance) within a person's own images."""
        print("\n--- Intra-Class Consistency ---")
        
        for name, vectors in vector_dict.items():
            if len(vectors) < 2:
                print(f"{name:<15}: N/A (Need 2+ images)")
                continue
            
            # Calculate all pairwise distances
            distances = []
            similarities = []
            for v1, v2 in combinations(vectors, 2):
                dist = np.linalg.norm(v1 - v2)
                distances.append(dist)
                similarity = np.dot(v1.flatten(), v2.flatten())
                similarities.append(similarity)
                
            avg_dist = sum(distances) / len(distances)
            avg_sim = sum(similarities) / len(similarities)
            print(f"{name:<15}: Distance {avg_dist:.4f} | Similarity {avg_sim:.4f}")

    @staticmethod
    def test_image_match(encoder: FaceEncoder, vector_dict: Dict[str, List[np.ndarray]], test_img_path: str):
        """Compares a test image against the database."""
        print(f"\n--- Analysis for Test Image: {test_img_path} ---")
        
        img = cv2.imread(test_img_path)
        if img is None:
            logger.error("Test image not found or unreadable.")
            return

        test_vector = encoder.get_embedding(img)
        if test_vector is None:
            logger.warning("No face detected in test image.")
            return

        for name, vectors in vector_dict.items():
            # Calculate distance to every vector we have for this person
            distances = [float(np.linalg.norm(test_vector - v)) for v in vectors]
            similarities = [float(np.dot(test_vector.flatten(), v.flatten())) for v in vectors]
            
            # Formatting for cleaner output
            # dist_str = ", ".join([f"{d:.3f}" for d in distances])
            min_dist = min(distances) if distances else 0
            max_sim = max(similarities) if similarities else 0
            
            print(f"{name:<15}: Min Dist: {min_dist:.3f} | Max Sim: {max_sim:.4f}")

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Initialize Encoder
    encoder = FaceEncoder(device=Config.DEVICE)
    manager = VectorManager(encoder)

    # 2. Process Directory and Save
    manager.process_directory(Config.FACES_DIR)
    manager.save_vectors(Config.OUTPUT_FILE)

    # 3. Run Analysis
    known_vectors = manager.get_vectors()
    
    if known_vectors:
        StatsAnalyzer.print_intra_class_distances(known_vectors)
        for test_img in os.listdir(Config.TEST_IMAGE_DIR):
            test_img_path = os.path.join(Config.TEST_IMAGE_DIR, test_img)
            StatsAnalyzer.test_image_match(encoder, known_vectors, test_img_path)
    else:
        logger.warning("No vectors available for analysis.")
