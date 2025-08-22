import cv2
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

class FaceDetector:
    def __init__(self):
        # Initialize the face cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Check if cascade loaded successfully
        if self.face_cascade.empty():
            raise Exception("Error loading face cascade classifier")
        
        self.results = []
        self.processed_count = 0
        self.error_count = 0
    
    def detect_faces_in_image(self, image_path):
        """
        Detect faces in a single image and return detection info
        """
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not read image: {os.path.basename(image_path)}")
                self.error_count += 1
                return None
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with optimized parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Get face information
            face_info = []
            for i, (x, y, w, h) in enumerate(faces):
                # Ensure face region is within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, img.shape[1] - x)
                h = min(h, img.shape[0] - y)
                
                if w > 0 and h > 0:
                    # Extract face region
                    face_region = img[y:y+h, x:x+w]
                    
                    # Calculate average color of the face region
                    avg_color = np.mean(face_region, axis=(0, 1))
                    
                    # Convert BGR to RGB for display
                    avg_color_rgb = [int(avg_color[2]), int(avg_color[1]), int(avg_color[0])]
                    
                    face_info.append({
                        'face_id': i + 1,
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'position': f"({x}, {y})",
                        'size': f"{w}x{h}",
                        'avg_color_rgb': avg_color_rgb,
                        'confidence_area': w * h
                    })
            
            self.processed_count += 1
            return {
                'filename': os.path.basename(image_path),
                'image_path': image_path,
                'total_faces': len(face_info),
                'faces': face_info,
                'image_size': f"{img.shape[1]}x{img.shape[0]}",
                'image_width': img.shape[1],
                'image_height': img.shape[0]
            }
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(image_path)}: {str(e)}")
            self.error_count += 1
            return None
    
    def process_directory(self, directory_path, show_progress=True, show_details=False):
        """
        Process all images in a directory with progress tracking
        """
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"❌ Directory '{directory_path}' not found.")
            return False
        
        # Find all image files
        print("🔍 Scanning for image files...")
        image_files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        if not image_files:
            print(f"❌ No image files found in '{directory_path}'")
            return False
        
        print(f"📁 Found {len(image_files)} image files")
        print(f"🚀 Starting face detection processing...")
        print("=" * 80)
        
        # Process images
        for i, image_file in enumerate(image_files, 1):
            if show_progress:
                print(f"Processing [{i}/{len(image_files)}]: {image_file.name}")
            
            result = self.detect_faces_in_image(str(image_file))
            if result:
                if show_details:
                    self.display_detection_result(result)
                else:
                    # Show brief progress
                    faces_found = result['total_faces']
                    if faces_found > 0:
                        print(f"  ✅ Found {faces_found} face(s)")
                    else:
                        print(f"  ⚪ No faces detected")
                
                self.results.append(result)
            
            # Progress update every 50 images
            if i % 50 == 0:
                print(f"📊 Progress: {i}/{len(image_files)} processed")
        
        print("=" * 80)
        self.show_summary()
        return True
    
    def display_detection_result(self, result):
        """
        Display detailed face detection results
        """
        print(f"📸 Image: {result['filename']}")
        print(f"   Size: {result['image_size']}")
        print(f"   Faces detected: {result['total_faces']}")
        
        if result['total_faces'] > 0:
            for face in result['faces']:
                print(f"   └── Face {face['face_id']}:")
                print(f"       Position: {face['position']}")
                print(f"       Size: {face['size']}")
                print(f"       Avg RGB Color: {face['avg_color_rgb']}")
                
                # Create a simple color representation
                r, g, b = face['avg_color_rgb']
                color_name = self.get_color_name(r, g, b)
                print(f"       Color description: {color_name}")
        
        print("-" * 60)
    
    def get_color_name(self, r, g, b):
        """
        Simple color classification based on RGB values
        """
        if r > 200 and g > 200 and b > 200:
            return "Light/Pale"
        elif r < 100 and g < 100 and b < 100:
            return "Dark"
        elif r > g and r > b:
            if r > 150:
                return "Reddish/Warm"
            else:
                return "Dark Reddish"
        elif g > r and g > b:
            return "Greenish"
        elif b > r and b > g:
            return "Bluish/Cool"
        elif r > 150 and g > 150:
            return "Yellowish/Warm"
        else:
            return "Medium tone"
    
    def show_summary(self):
        """
        Display processing summary with statistics
        """
        total_images = len(self.results)
        images_with_faces = sum(1 for r in self.results if r['total_faces'] > 0)
        images_no_faces = total_images - images_with_faces
        total_faces = sum(r['total_faces'] for r in self.results)
        
        print("📊 PROCESSING SUMMARY")
        print("=" * 50)
        print(f"✅ Total images processed: {self.processed_count}")
        print(f"❌ Images with errors: {self.error_count}")
        print(f"👥 Images with faces: {images_with_faces}")
        print(f"⚪ Images without faces: {images_no_faces}")
        print(f"🎯 Total faces detected: {total_faces}")
        
        if total_images > 0:
            avg_faces = total_faces / total_images
            print(f"📈 Average faces per image: {avg_faces:.2f}")
            success_rate = (images_with_faces / total_images) * 100
            print(f"📊 Face detection success rate: {success_rate:.1f}%")
        
        print("=" * 50)
    
    def save_results_to_csv(self, output_path="face_detection_results.csv"):
        """
        Save detection results to a CSV file
        """
        if not self.results:
            print("❌ No results to save.")
            return False
        
        print("💾 Preparing CSV data...")
        rows = []
        
        for result in self.results:
            base_row = {
                'filename': result['filename'],
                'image_path': result['image_path'],
                'image_width': result['image_width'],
                'image_height': result['image_height'],
                'image_size': result['image_size'],
                'total_faces': result['total_faces']
            }
            
            if result['total_faces'] == 0:
                # No faces found
                row = base_row.copy()
                row.update({
                    'face_id': None,
                    'face_x': None,
                    'face_y': None,
                    'face_width': None,
                    'face_height': None,
                    'position': None,
                    'size': None,
                    'avg_color_rgb': None,
                    'color_r': None,
                    'color_g': None,
                    'color_b': None,
                    'color_description': None
                })
                rows.append(row)
            else:
                # Add row for each face
                for face in result['faces']:
                    row = base_row.copy()
                    r, g, b = face['avg_color_rgb']
                    row.update({
                        'face_id': face['face_id'],
                        'face_x': face['x'],
                        'face_y': face['y'],
                        'face_width': face['width'],
                        'face_height': face['height'],
                        'position': face['position'],
                        'size': face['size'],
                        'avg_color_rgb': str(face['avg_color_rgb']),
                        'color_r': r,
                        'color_g': g,
                        'color_b': b,
                        'color_description': self.get_color_name(r, g, b)
                    })
                    rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        try:
            df.to_csv(output_path, index=False)
            print(f"✅ Results saved to: {output_path}")
            print(f"📄 Total rows in CSV: {len(df)}")
            return True
        except Exception as e:
            print(f"❌ Error saving CSV: {str(e)}")
            return False
    
    def get_face_statistics(self):
        """
        Get detailed statistics about detected faces
        """
        if not self.results:
            return None
        
        stats = {
            'total_images': len(self.results),
            'images_with_faces': 0,
            'total_faces': 0,
            'face_sizes': [],
            'colors': {'light': 0, 'dark': 0, 'medium': 0, 'warm': 0, 'cool': 0}
        }
        
        for result in self.results:
            if result['total_faces'] > 0:
                stats['images_with_faces'] += 1
                stats['total_faces'] += result['total_faces']
                
                for face in result['faces']:
                    stats['face_sizes'].append(face['confidence_area'])
                    
                    # Color classification
                    r, g, b = face['avg_color_rgb']
                    color_desc = self.get_color_name(r, g, b).lower()
                    
                    if 'light' in color_desc or 'pale' in color_desc:
                        stats['colors']['light'] += 1
                    elif 'dark' in color_desc:
                        stats['colors']['dark'] += 1
                    elif 'warm' in color_desc or 'reddish' in color_desc:
                        stats['colors']['warm'] += 1
                    elif 'cool' in color_desc or 'bluish' in color_desc:
                        stats['colors']['cool'] += 1
                    else:
                        stats['colors']['medium'] += 1
        
        return stats

def main():
    """
    Main function to run face detection on your image folder
    """
    print("🎯 Face Detection for Image Folder")
    print("=" * 50)
    
    # Initialize detector
    try:
        detector = FaceDetector()
        print("✅ Face detector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize face detector: {str(e)}")
        print("\n💡 Make sure OpenCV is properly installed:")
        print("pip install opencv-python")
        return
    
    # Your image folder path
    default_path = r"E:\My Training -AI&ML\Humans"
    
    print(f"\n📁 Default image folder: {default_path}")
    custom_path = input("Enter different path or press Enter to use default: ").strip()
    
    if custom_path:
        image_folder = custom_path
    else:
        image_folder = default_path
    
    print(f"\n🔍 Processing images from: {image_folder}")
    
    # Ask for processing options
    print("\n⚙️  Processing Options:")
    print("1. Quick mode (show progress only)")
    print("2. Detailed mode (show face details for each image)")
    
    mode = input("Choose mode (1 or 2, default=1): ").strip()
    show_details = (mode == "2")
    
    # Process the directory
    start_time = datetime.now()
    
    success = detector.process_directory(
        image_folder, 
        show_progress=True, 
        show_details=show_details
    )
    
    if not success:
        return
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
    
    # Show detailed statistics
    stats = detector.get_face_statistics()
    if stats and stats['total_faces'] > 0:
        print("\n📈 DETAILED STATISTICS")
        print("=" * 40)
        avg_faces = stats['total_faces'] / stats['total_images']
        print(f"Average faces per image: {avg_faces:.2f}")
        
        if stats['face_sizes']:
            avg_face_size = np.mean(stats['face_sizes'])
            print(f"Average face size: {avg_face_size:.0f} pixels²")
        
        print("\nColor distribution:")
        for color, count in stats['colors'].items():
            if count > 0:
                percentage = (count / stats['total_faces']) * 100
                print(f"  {color.capitalize()}: {count} faces ({percentage:.1f}%)")
    
    # Offer to save results
    if detector.results:
        save_choice = input("\n💾 Save results to CSV file? (y/n, default=y): ").strip().lower()
        if save_choice != 'n':
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"face_detection_results_{timestamp}.csv"
            detector.save_results_to_csv(output_file)
    
    print("\n🎉 Face detection completed!")

if __name__ == "__main__":
    # Check if required packages are available
    try:
        import cv2
        import pandas as pd
        import numpy as np
        print("✅ All required packages are available")
        main()
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("\n💡 Install required packages with:")
        print("pip install opencv-python pandas numpy")
        input("\nPress Enter to exit...")