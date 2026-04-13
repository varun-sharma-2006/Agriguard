import torch
import torch.nn as nn
from src.models.multi_modal_cnn import MultiModalCNN, AgriDataset
import numpy as np

class DiseasePredictor:
    def __init__(self, model_path='models/agriguard_working_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the saved artifact dictionary
        # Needs to map location to CPU incase trained on GPU and no GPU is available
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.label_encoder = checkpoint['label_encoder']
        self.features_scaler = checkpoint['features_scaler']
        hyperparameters = checkpoint['hyperparameters']
        
        # Instantiate Model
        self.model = MultiModalCNN(
            num_classes=hyperparameters.get('num_classes', len(self.label_encoder.classes_)),
            learning_rate=hyperparameters.get('learning_rate', 1e-3)
        )
        
        # Load states and swap to eval mode
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_features(self, ndvi, evi, savi, rep, temp_max, temp_min, humidity, rainfall):
        """Standardizes input and reshapes into Torch modalities."""
        
        # Order must exactly match: ['NDVI', 'EVI', 'SAVI', 'REP', 'temp_max', 'temp_min', 'humidity', 'rainfall']
        raw_features = np.array([[ndvi, evi, savi, rep, temp_max, temp_min, humidity, rainfall]])
        
        # Scale with the training scaler
        scaled_features = self.features_scaler.transform(raw_features)[0]
        
        # Extract modalities
        spectral_arr = scaled_features[0:4]
        weather_arr = scaled_features[4:8]
        
        spectral_tensor = torch.tensor(spectral_arr, dtype=torch.float32)
        weather_tensor = torch.tensor(weather_arr, dtype=torch.float32)
        
        # The create_spatial_representation expects a 'row' with named dict access + spectral_values
        # But looking at AgriDataset's src code, it only uses spectral_values[0] for NDVI to create spatial noise!
        # So I can pass a dummy row format or mock the AgriDataset method.
        # Let's instantiate a dummy Dataset method specifically for its spatial builder.
        
        dataset_mock = AgriDataset.__new__(AgriDataset)
        spatial_patch = dataset_mock.create_spatial_representation(row=None, spectral_values=spectral_tensor)
        
        # Expand dims for batch size of 1
        return (
            spatial_patch.unsqueeze(0).to(self.device), 
            spectral_tensor.unsqueeze(0).to(self.device), 
            weather_tensor.unsqueeze(0).to(self.device)
        )

    def predict(self, ndvi, evi, savi, rep, temp_max, temp_min, humidity, rainfall):
        """Full inference pipeline."""
        
        spatial, spectral, weather = self.preprocess_features(
            ndvi, evi, savi, rep, temp_max, temp_min, humidity, rainfall
        )
        
        with torch.no_grad():
            output = self.model(spatial, spectral, weather)
            probs = torch.softmax(output, dim=1).squeeze()
            confidence, predicted_idx = torch.max(probs, dim=0)
            
            prediction_class = self.label_encoder.inverse_transform([predicted_idx.item()])[0]
            
            # Map probabilities to classes
            class_probs = {self.label_encoder.classes_[i]: probs[i].item() for i in range(len(self.label_encoder.classes_))}
            
            return {
                'predicted_class': prediction_class,
                'confidence': confidence.item(),
                'probabilities': class_probs,
                'disease_risk_score': 1.0 - class_probs.get('healthy', 0.0) # Risk is inverse of healthy
            }
