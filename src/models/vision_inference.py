import torch
from torchvision import transforms, models
from PIL import Image
import io
import torch.nn as nn

class ImagePredictor:
    def __init__(self, model_path='models/vision_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.class_names = checkpoint.get('class_names', ['Unknown'])
            hyperparameters = checkpoint.get('hyperparameters', {})
            num_classes = hyperparameters.get('num_classes', len(self.class_names))
            state_dict = checkpoint.get('model_state_dict')
            
            # Recreate Backbone
            self.model = models.resnet18()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_ftrs, num_classes)
            )
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.ready = True
        except Exception as e:
            print(f"Warning: Could not load vision model cleanly: {e}")
            self.ready = False
            self.class_names = ['healthy', 'diseased', 'stressed'] # Fallback
            
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict_image(self, image_bytes):
        """Processes uploaded image bytes and runs ResNet inference."""
        if not self.ready:
            return {
                'predicted_class': 'model_not_trained',
                'confidence': 0.0,
                'probabilities': {c: 0.0 for c in self.class_names}
            }
            
        # Parse Bytes into PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Transform and add batch dimension
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1).squeeze()
            
            confidence, predicted_idx = torch.max(probs, dim=0)
            prediction_class = self.class_names[predicted_idx.item()]
            
            # Map probabilities
            class_probs = {self.class_names[i]: probs[i].item() for i in range(len(self.class_names))}
            
            return {
                'predicted_class': prediction_class,
                'confidence': confidence.item(),
                'probabilities': class_probs
            }
