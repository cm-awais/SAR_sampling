import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import json
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torchvision.transforms as transforms
import csv
# from m_models import MinorityClassClassifier
from m_models import *
from collections import Counter

# transform_train = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
# ])

def get_classification_metrics(true_labels, predicted_labels):
    """
    Compute accuracy, precision, recall, and F1-score for each class.

    Args:
        true_labels (list): True class labels.
        predicted_labels (list): Predicted class labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score per class.
    """
    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Accuracy per class
    class_accuracies = (conf_matrix.diagonal() / conf_matrix.sum(axis=1)).tolist()

    # Precision, Recall, F1-score per class
    precision = precision_score(true_labels, predicted_labels, average=None).tolist()
    recall = recall_score(true_labels, predicted_labels, average=None).tolist()
    f1 = f1_score(true_labels, predicted_labels, average=None).tolist()

    # Prepare dictionary
    metrics = {
        "accuracy": class_accuracies,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    metrics_list = []
    for j in range(len(class_accuracies)):
        metrics_list.extend([class_accuracies[j], precision[j], recall[j], f1[j]])

    # metrics_list = [class_accuracies, precision, recall, f1]

    return metrics, metrics_list

def cal_scores(targets, predictions, check=False):
  if check:
    true_labels = [int(t.item()) for t in targets]  # Extract integer values
    predicted_labels = [int(p.item()) for p in predictions]

  scores = []
  # Generate classification report
  report, metrics_list = get_classification_metrics(true_labels, predicted_labels)

  overall_accuracy = round(accuracy_score(true_labels, predicted_labels), 4)* 100

  # Calculate overall precision, recall, and F1-score (weighted average)
  overall_precision = round(precision_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  overall_recall = round(recall_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  overall_f1 = round(f1_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  scores.extend([overall_accuracy, overall_precision, overall_recall, overall_f1])
  scores.extend(metrics_list)

  return {
      'class_metrics' : report,
      'overall_accuracy': overall_accuracy,
      'overall_precision': overall_precision,
      'overall_recall': overall_recall,
      'overall_f1': overall_f1
  }, scores


# Evaluation function 
def evaluate_model(model, data_loader, feature_extractor, device):
    model.eval()
    model.to(device)
    saving_string = ""
    correct = 0
    total = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            # print(data.shape, target.shape)
            features = data
            if feature_extractor is not None:
                features = feature_extractor(data)  # uncomment
            
            output = model(features)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted)
            targets.extend(target)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    saving_string += f"Accuracy: {accuracy:.2f}% \n"
    # print(f"Accuracy: {accuracy:.2f}%")
    print()
    dicrt, scores = cal_scores(predictions=predictions, targets=targets, check=True)
    # print(dicrt)
    saving_string += json.dumps(dicrt, indent=4)
    return saving_string, scores, dicrt

def print_data_ratios(dataloader):
    """
    Prints the class ratios from a given PyTorch DataLoader.

    Args:
        dataloader (DataLoader): The PyTorch DataLoader to analyze.
    """
    # Collect all labels from the dataloader
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.numpy())  # Convert labels tensor to numpy and extend the list

    # Count the occurrences of each class
    class_counts = Counter(all_labels)
    total_samples = sum(class_counts.values())

    # Calculate and print class ratios
    class_ratios = {cls: count / total_samples for cls, count in class_counts.items()}
    print("Updated Class Ratios:", class_ratios)

def load_data(address, classes_l, transform_train, batch_size=32, train=True):
  new_class_to_idx = {}
  dataset = ImageFolder(root=address, transform=transform_train)
  # Load Fusar dataset
  for id, class_n in enumerate(classes_l):
    new_class_to_idx[class_n] = id

  dataset.class_to_idx = new_class_to_idx

  # Create a dictionary of class names
  class_names = {i: classname for i, classname in enumerate(dataset.classes)}

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2,  # Experiment with different values as recommended above
                            # pin_memory=False, # if torch.cuda.is_available() else False,
                            persistent_workers=True)
  print("Top classes indices:", class_names)

  return data_loader

def load_data_m2m(address, classes_l, transform_train, batch_size=32, dataset_size=None, train=True):
    """
    Load a dataset, calculate class ratios dynamically, and create a DataLoader with oversampling.

    Args:
        address (str): Path to the dataset.
        transform_train (torchvision.transforms): Transformations to apply to the dataset.
        batch_size (int): Batch size for the DataLoader.
        dataset_size (int or None): Total number of samples in the dataset. If None, it is computed dynamically.
        train (bool): Whether the dataset is for training or not.

    Returns:
        DataLoader: A PyTorch DataLoader with oversampling.
    """
    # Load the dataset
    new_class_to_idx = {}
    dataset = ImageFolder(root=address, transform=transform_train)

    for id, class_n in enumerate(classes_l):
        new_class_to_idx[class_n] = id

    dataset.class_to_idx = new_class_to_idx
    
    # Calculate class distribution
    class_counts = Counter([label for _, label in dataset])  # Count samples per class
    total_samples = sum(class_counts.values())
    class_ratios = {cls: count / total_samples for cls, count in class_counts.items()}
    
    print("Class distribution:", class_counts)
    print("Class ratios:", class_ratios)

    # Calculate the number of samples per class based on ratios
    if dataset_size is None:
        dataset_size = total_samples  # Use the dataset's total size if not provided

    num_sample_per_class = [int(class_ratios[cls] * dataset_size) for cls in range(len(class_counts))]
    print("Calculated samples per class:", num_sample_per_class)

    # Create oversampled weights
    def get_oversampled_data(dataset, num_sample_per_class):
        length = len(dataset)
        num_sample_per_class = list(num_sample_per_class)
        num_samples = list(num_sample_per_class)

        selected_list = []
        indices = list(range(length))
        for i in range(length):
            _, label = dataset.__getitem__(indices[i])
            if num_sample_per_class[label] > 0:
                selected_list.append(1 / num_samples[label])
                num_sample_per_class[label] -= 1

        return selected_list

    oversampled_weights = get_oversampled_data(dataset, num_sample_per_class)
    print("Generated oversampling weights:", oversampled_weights[:10], "...")

    # Use WeightedRandomSampler for balanced sampling
    sampler = WeightedRandomSampler(oversampled_weights, len(oversampled_weights))

    # Create the DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, persistent_workers=True)

    print_data_ratios(data_loader)

    print("DataLoader created with oversampling.")
    return data_loader

# weights initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_loaders_b(path, data, classes_l, batch_size, transform_train, m2m=False):
    loaders = []
    # print(classes_l)
    if m2m:
        loaders.append(load_data_m2m(path + "/train", classes_l=classes_l, transform_train=transform_train, batch_size=batch_size))
    else:
        loaders.append(load_data(path + "/train", classes_l=classes_l, transform_train=transform_train, batch_size=batch_size))
    
    loaders.append(load_data(path + "/val", classes_l=classes_l, transform_train=transform_train, batch_size=batch_size))
    loaders.append(load_data(path + "/test", classes_l=classes_l, transform_train=transform_train, batch_size=batch_size))
    
    return loaders

def calculate_centroids(features, labels, num_classes):
    centroids = []
    for c in range(num_classes):
        class_features = features[labels == c]
        centroids.append(class_features.mean(dim=0))
    return centroids

def generate_synthetic_samples_with_rejection(features, labels, centroids, target_class, alpha=0.1, max_samples=500, similarity_threshold=0.8, classifier=None):
    """
    Generate synthetic samples with rejection criteria.
    
    Args:
        features (torch.Tensor): Feature vectors of the dataset, shape [num_samples, feature_dim].
        labels (torch.Tensor): Corresponding labels, shape [num_samples].
        centroids (list of torch.Tensor): List of class centroids.
        target_class (int): Target minority class to generate synthetic samples for.
        alpha (float): Translation factor for feature interpolation.
        max_samples (int): Maximum number of synthetic samples to generate.
        similarity_threshold (float): Maximum similarity to existing samples in the target class.
        classifier (torch.nn.Module): Optional, a classifier to evaluate synthetic sample quality.
    
    Returns:
        synthetic_features (torch.Tensor): Synthetic feature vectors.
        synthetic_labels (torch.Tensor): Corresponding labels for synthetic features.
    """
    synthetic_features = []
    synthetic_labels = []
    
    # Collect existing features of the target class
    target_class_features = features[labels == target_class]
    
    for feature, label in zip(features, labels):
        if label != target_class:  # Only translate from majority classes
            # Generate synthetic feature
            translated_feature = feature + alpha * (centroids[target_class] - feature)
            
            # Rejection criterion 1: Avoid redundancy using similarity
            similarity = F.cosine_similarity(translated_feature.unsqueeze(0), target_class_features).max().item()
            if similarity > similarity_threshold:
                # print(similarity, label, target_class)
                continue  # Skip overly similar samples
            
            # Rejection criterion 2: Model confidence (if classifier is provided)
            if classifier is not None:
                classifier.eval()  # Ensure the classifier is in evaluation mode
                with torch.no_grad():
                    confidence = F.softmax(classifier(translated_feature.unsqueeze(0)), dim=1)[0, target_class].item()
                if confidence > 0.9:  # Skip if classifier is already confident
                    continue
            
            # Add synthetic sample
            synthetic_features.append(translated_feature)
            synthetic_labels.append(target_class)
            
            # Stop if we reach the max_samples limit
            if len(synthetic_features) >= max_samples:
                print(label, len(synthetic_features))
                break
    
    if len(synthetic_features) == 0:  # Handle case where no samples are accepted
        return torch.empty(0, features.size(1)), torch.empty(0, dtype=torch.long)
    
    return torch.stack(synthetic_features), torch.tensor(synthetic_labels)

def m2m_generate_synthetic_samples(features, labels, centroids, target_class, alpha=0.1, max_samples=500, rejection_threshold=0.5):
    """
    Generate synthetic samples for a minority class using M2m strategy with rejection criteria.
    
    Args:
        features (torch.Tensor): Feature vectors of shape [num_samples, feature_dim].
        labels (torch.Tensor): Corresponding labels of shape [num_samples].
        centroids (list): List of precomputed centroids for each class.
        target_class (int): Minority class for which synthetic samples are generated.
        alpha (float): Translation factor to control the strength of the perturbation.
        max_samples (int): Maximum number of synthetic samples to generate.
        rejection_threshold (float): Threshold for rejecting redundant synthetic samples.
        
    Returns:
        synthetic_features (torch.Tensor): Synthetic feature vectors.
        synthetic_labels (torch.Tensor): Corresponding labels for synthetic features.
    """
    synthetic_features = []
    synthetic_labels = []

    # Extract the centroid of the target minority class
    target_centroid = centroids[target_class]
    
    # Calculate distances to reject redundant samples
    def is_valid_sample(new_sample, existing_samples, threshold):
        if len(existing_samples) == 0:
            return True  # First sample is always valid
        distances = torch.norm(existing_samples - new_sample, dim=1)  # Euclidean distance
        min_distance = torch.min(distances).item()
        return min_distance > threshold  # Reject if too close to existing samples

    # Generate synthetic samples
    existing_synthetic_features = []
    for feature, label in zip(features, labels):
        if label != target_class:  # Translate from majority classes only
            # Translate feature toward the target class centroid
            translated_feature = feature + alpha * (target_centroid - feature)

            # Apply rejection criteria
            if is_valid_sample(translated_feature, torch.stack(existing_synthetic_features) if existing_synthetic_features else [], rejection_threshold):
                synthetic_features.append(translated_feature)
                synthetic_labels.append(target_class)
                existing_synthetic_features.append(translated_feature)

                # Stop if max_samples is reached
                if len(synthetic_features) >= max_samples:
                    print(target_class, len(synthetic_features))
                    break
    

    if len(synthetic_features) == 0:
        return torch.empty(0, features.size(1)), torch.empty(0, dtype=torch.long)
    
    return torch.stack(synthetic_features), torch.tensor(synthetic_labels)



def m2m_creation(train_loader, feature_extractor, classes, minority_value, device, samp_met=0):
    # Extract features and labels
    features = []
    labels = []
    synthetic_features = []
    synthetic_labels = []

    for id, (inputs, targets) in enumerate(train_loader):
        # print(id)
        with torch.no_grad():
            inputs = inputs.cuda()  # Move to GPU if available
            w_feat = feature_extractor(inputs)
            # print(w_feat.shape)
            extracted_features = w_feat #.mean(dim=(2, 3))  # Global average pooling
            features.append(extracted_features)
            labels.append(targets)
    features = torch.cat(features, dim=0)  # Combine all feature tensors
    labels = torch.cat(labels, dim=0)      # Combine all label tensors

    # input_size = features.size(1) # baseline

    # o_dataset = TensorDataset(features, labels) # baseline
    # o_loader = DataLoader(o_dataset, batch_size=32, shuffle=True) # baseline

    # return input_size, o_loader

    # print(id)

    num_classes = classes
    
    centroids = calculate_centroids(features, labels, num_classes)
    # print("Centroids: ",centroids)

    # Identify minority classes (e.g., classes with fewer than a threshold number of samples)
    class_counts = torch.bincount(labels)
    print("class_counts", class_counts)
    minority_classes = torch.where(class_counts < minority_value)[0]  # Adjust threshold as needed
    print("Monority Classes: ",minority_classes, minority_value)

    for target_class in minority_classes:
        if samp_met==0:
            features_aug, labels_aug = generate_synthetic_samples_with_rejection(features, labels, centroids, target_class)
        else:
            features_aug, labels_aug = m2m_generate_synthetic_samples(features, labels, centroids, target_class)
        synthetic_features.append(features_aug)
        synthetic_labels.append(labels_aug)

    synthetic_features = torch.cat(synthetic_features, dim=0)
    synthetic_labels = torch.cat(synthetic_labels, dim=0)

    print("Device: ",synthetic_features.shape, features.shape)

    # Combine original and synthetic data
    augmented_features = torch.cat([features, synthetic_features], dim=0)
    augmented_labels = torch.cat([labels, synthetic_labels], dim=0)

    input_size = augmented_features.size(1)  # Size of feature vector uncomment
    # print(augmented_features.shape)
    # print(synthetic_features.shape)
    # print(features.shape)

    # Create DataLoader
    augmented_dataset = TensorDataset(augmented_features, augmented_labels)

    augmented_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
    # input_size = augmented_features.size(1)
    
    return input_size, augmented_loader
def train_save_1(data_dict, l_rate, device, epochs, lf):
    sampling_methods = ["sat1", "m2m"]
    for ij in range(2):
        csv_res = []
        csv1 = []
        results = ""
        ds = sampling_methods[ij] + "_"
        for dataset_name, dataset_loader in data_dict.items():
            print("Training on ", dataset_name)
            ds += dataset_name + "_"
            results += "Training on " + dataset_name + "\n"
            classes = dataset_loader[1]
            
            minority_value = dataset_loader[2]
            models = {
            # "VIT": vit(classes, frozen=True),
            "Fine_VGG": FineTunedVGG(classes),
            "Fine_Resnet": FineTunedResNet(classes)
            }
            # mix_path = "mix_5"
            train_loader_m = dataset_loader[0][0]
            val_loader_m = dataset_loader[0][1]
            test_loader_m = dataset_loader[0][2]
            for model_name, model in models.items():
                print("Training using :" , model_name)
                feature_extractor = None
                ds += model_name
                if "VIT" in model_name:
                    feature_extractor = model
                    # print("VIT", model_name)
                else:
                    feature_extractor = model.features
                    # print("Non Vit", model_name)
                feature_extractor.to(device)
                # if baseline:
                #     train_loader_n = train_loader_m
                # else:
                #     ip, train_loader_n = m2m_creation(train_loader=train_loader_m, feature_extractor=feature_extractor, classes=classes, minority_value=minority_value, device=device, samp_met=ij)
                ip, train_loader_n = m2m_creation(train_loader=train_loader_m, feature_extractor=feature_extractor, classes=classes, minority_value=minority_value, device=device, samp_met=ij)
                
                results += "Training using "+model_name + "\n"
                    
                n_model = MinorityClassClassifier(input_size=ip, classes=classes, mod=model_name)
                n_model.apply(initialize_weights)
                optimizer = optim.Adam(n_model.parameters(), lr=l_rate)
                n_model.to(device)
                # Training loop
                for epoch in range(epochs):
                    n_model.train()
                    # print(epoch)
                    for batch_idx, data in enumerate(train_loader_m):
                        data, target = data[0].to(device), data[1].to(device)

                        optimizer.zero_grad()

                        output = n_model(data)

                        loss = lf(output, target)
                        loss.backward()
                        optimizer.step()
                    # else:
                    print("Validation: ", epoch, dataset_name)
                    results += f"Validation {model_name} on {dataset_name} \n"
                    str_results, _, _ = evaluate_model(n_model, val_loader_m, feature_extractor=feature_extractor, device=device)
                    results += str_results
                print("Testing: ", dataset_name)
                results += f"Testing {model_name} on {dataset_name} \n"
                str_results, csv_scores, dict_o = evaluate_model(n_model, test_loader_m, feature_extractor=feature_extractor, device=device)
                print(dict_o)
                results += str_results
                csv_res.append(csv_scores)
                csv1.append(csv_scores)
                
            # save_model(n_model, save_dir, model_filename)

        # Open the file in write mode ("w") and write the string to it
        with open(f"scores/sampling_results_{ds}_64.txt", "w") as f:
            f.write(results)

        fields = ["Accuracy", "Precision", "Recall", "F1"] * (classes +1)

        with open(f"scores/sampling_results_{ds}_64.csv", 'w') as f:
                write = csv.writer(f)
                write.writerow(fields)
                write.writerows(csv1)

def train_save_b(data_dict, l_rate, device, epochs, lf, m2m=False):
    sampling_methods = ["M2MM", "base"]
    for ij in range(1):
        csv_res = []
        csv1 = []
        results = ""
        ds = ""
        if m2m:
            ds += "m2m" + "_"
        else:
            ds += "baseline" + "_"
        for dataset_name, dataset_loader in data_dict.items():
            print("Training on ", dataset_name)
            ds += dataset_name + "_"
            results += "Training on " + dataset_name + "\n"
            classes = dataset_loader[1]
            
            minority_value = dataset_loader[2]
            models = {
            # "VIT": vit(classes, frozen=True),
            "Fine_VGG": FineTunedVGG(classes),
            "Fine_Resnet": FineTunedResNet(classes)
            }
            # mix_path = "mix_5"
            train_loader_m = dataset_loader[0][0]
            val_loader_m = dataset_loader[0][1]
            test_loader_m = dataset_loader[0][2]
            for model_name, model in models.items():
                print("Training using :" , model_name)
                feature_extractor = None
                ds += model_name
                # if "VIT" in model_name:
                #     feature_extractor = model
                #     # print("VIT", model_name)
                # else:
                #     feature_extractor = model.features
                #     # print("Non Vit", model_name)
                # feature_extractor.to(device)
                # if baseline:
                #     train_loader_n = train_loader_m
                # else:
                #     ip, train_loader_n = m2m_creation(train_loader=train_loader_m, feature_extractor=feature_extractor, classes=classes, minority_value=minority_value, device=device, samp_met=ij)
                # ip, train_loader_n = m2m_creation(train_loader=train_loader_m, feature_extractor=feature_extractor, classes=classes, minority_value=minority_value, device=device, samp_met=ij)
                
                results += "Training using "+model_name + "\n"
                    
                # n_model = MinorityClassClassifier(input_size=ip, classes=classes, mod=model_name)
                n_model = model
                n_model.apply(initialize_weights)
                optimizer = optim.Adam(n_model.parameters(), lr=l_rate)
                n_model.to(device)
                # Training loop
                for epoch in range(epochs):
                    n_model.train()
                    # print(epoch)
                    for batch_idx, data in enumerate(train_loader_m):
                        data, target = data[0].to(device), data[1].to(device)

                        optimizer.zero_grad()

                        output = n_model(data)

                        loss = lf(output, target)
                        loss.backward()
                        optimizer.step()
                    # else:
                    print("Validation: ", epoch, dataset_name)
                    results += f"Validation {model_name} on {dataset_name} \n"
                    str_results, _, _ = evaluate_model(n_model, val_loader_m, feature_extractor=feature_extractor, device=device)
                    results += str_results
                print("Testing: ", dataset_name)
                results += f"Testing {model_name} on {dataset_name} \n"
                str_results, csv_scores, dict_o = evaluate_model(n_model, test_loader_m, feature_extractor=feature_extractor, device=device)
                print(dict_o)
                results += str_results
                csv_res.append(csv_scores)
                csv1.append(csv_scores)
                
            # save_model(n_model, save_dir, model_filename)

        # Open the file in write mode ("w") and write the string to it
        # with open(f"scores/base_sampling_results_{ds}_64.txt", "w") as f:
        #     f.write(results)

        fields = ["Accuracy", "Precision", "Recall", "F1"] * (classes +1)

        with open(f"scores/original_sampling_{ds}_64.csv", 'w') as f:
                write = csv.writer(f)
                write.writerow(fields)
                write.writerows(csv1)
