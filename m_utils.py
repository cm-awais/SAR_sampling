import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
import json
import os
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torchvision.transforms as transforms
import csv
from m_models import MinorityClassClassifier

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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
def evaluate_model(model, data_loader, device):
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
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted)
            targets.extend(target)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    saving_string += f"Accuracy: {accuracy:.2f}% \n"
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    dicrt, scores = cal_scores(predictions=predictions, targets=targets, check=True)
    print(dicrt)
    saving_string += json.dumps(dicrt, indent=4)
    return saving_string, scores

def load_data(address, batch_size=32, train=True):
  # Load Fusar dataset
  if train:
    dataset = ImageFolder(root=address, transform=transform_train)
  else: 
    dataset = ImageFolder(root=address, transform=transform_train)

  # Create a dictionary of class names
  class_names = {i: classname for i, classname in enumerate(dataset.classes)}

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2,  # Experiment with different values as recommended above
                            # pin_memory=False, # if torch.cuda.is_available() else False,
                            persistent_workers=True)
  print("Top classes indices:", class_names)

  return data_loader


# Function to calculate per-class validation loss
def compute_classwise_validation_loss(model, val_loader, loss_fn, num_classes, device):
    model.eval()
    class_loss = torch.zeros(num_classes, device=device)
    class_count = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss_per_sample = loss_fn(outputs, targets)  # Individual losses
            for i in range(num_classes):
                mask = targets == i  # Select samples for class i
                class_loss[i] += loss_per_sample[mask].sum()
                class_count[i] += mask.sum()

    # Avoid division by zero
    class_loss /= (class_count + 1e-5)
    return class_loss

# weights initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Function to compute weights
def compute_weights_with_zeros(ratios):
    ratios = torch.tensor(ratios, dtype=torch.float32)
    # Replace zeros with a very small value to avoid division errors
    safe_ratios = torch.where(ratios == 0, torch.tensor(1e-5), ratios)
    weights = 1.0 / safe_ratios  # Inverse of ratios
    normalized_weights = weights / weights.sum()  # Normalize weights
    return normalized_weights.tolist()

def compute_weights_baseline(ratios):
    # Calculate weights as inverse of ratios
    weights = 1.0 / ratios

    # Normalize weights (optional, ensures the weights sum to 1)
    normalized_weights = weights / weights.sum()

    return normalized_weights

def compute_weights_levels(data):
    """data=0: opensar, data=1 fusar"""
    if data == 0:
        # Ratios for OpenSAR
        l1 = [67.02181634, 32.97818366, 0, 0, 0, 0]
        l2 = [36.58536585, 29.57317073, 17.07317073, 16.76829268, 0, 0]
        l3 = [29.27927928, 22.97297297, 12.61261261, 12.38738739, 11.71171171, 11.03603604]
    else:
        # Ratios for Fusar
        l1 = [58.18908123, 41.81091877, 0, 0, 0, 0, 0, 0, 0]
        l2 = [30.23255814, 29.71576227, 20.93023256, 19.12144703, 0, 0, 0, 0, 0]
        l3 = [24.06181015, 12.80353201, 11.9205298, 9.713024283, 9.933774834, 8.609271523, 9.492273731, 7.06401766, 6.401766004]

    # Compute weights for each level
    weights_l1 = compute_weights_with_zeros(l1)
    weights_l2 = compute_weights_with_zeros(l2)
    weights_l3 = compute_weights_with_zeros(l3)

    return [weights_l1, weights_l2, weights_l3]


def get_loaders(dir, data):
    new_class_to_idx = {}
    if data==0:
        new_class_to_idx = {'Cargo': 0, 'Tanker': 1, 'Dredging': 2, 'Fishing': 3, 'Passenger': 4, 'Tug': 5}
    else:
        new_class_to_idx = {'Cargo': 0, 'Fishing': 1, 'Bluk': 2, 'Dredging': 3, 'Container': 4, 'Tanker': 5, 'GeneralCargo': 6, 'Passenger': 7, 'Tug': 8} #fusar


    # Load datasets
    curriculum_data_dir = dir
    easy_dataset = ImageFolder(os.path.join(curriculum_data_dir, "easy"), transform=transform_train)
    moderate_dataset = ImageFolder(os.path.join(curriculum_data_dir, "moderate"), transform=transform_train)
    hard_dataset = ImageFolder(os.path.join(curriculum_data_dir, "hard"), transform=transform_train)
    validation_dataset = ImageFolder(os.path.join(curriculum_data_dir, "validation"), transform=transform_train)
    test_dataset = ImageFolder(os.path.join(curriculum_data_dir, "test"), transform=transform_train)

    # new_class_to_idx = {'Cargo': 0, 'Tanker': 1, 'Dredging': 2, 'Fishing': 3, 'Passenger': 4, 'Tug': 5} # opensar
    # new_class_to_idx = {'Cargo': 0, 'Fishing': 1, 'Bluk': 2, 'Dredging': 3, 'Container': 4, 'Tanker': 5, 'GeneralCargo': 6, 'Passenger': 7, 'Tug': 8} #fusar


    easy_dataset.class_to_idx = new_class_to_idx
    moderate_dataset.class_to_idx = new_class_to_idx
    hard_dataset.class_to_idx = new_class_to_idx
    validation_dataset.class_to_idx = new_class_to_idx
    test_dataset.class_to_idx = new_class_to_idx

    # Create DataLoaders
    batch_size = 32
    easy_loader = DataLoader(easy_dataset, batch_size=batch_size, shuffle=True)
    moderate_loader = DataLoader(moderate_dataset, batch_size=batch_size, shuffle=True)
    hard_loader = DataLoader(hard_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return [easy_loader, moderate_loader, hard_loader, validation_loader, test_loader]

print("DataLoaders created successfully.")

print("Loading Data")

def get_loaders_b(data, batch_size):
    loaders = []
    fusar_path = "c_fusar_splited"
    opensar_path = "c_opensar_splited"
    classes = 0
    models = {}
    if data == 0:
        loaders.append(load_data(opensar_path + "/train", batch_size=batch_size))
        loaders.append(load_data(opensar_path + "/val", batch_size=batch_size))
        loaders.append(load_data(opensar_path + "/test", batch_size=batch_size))
    else:
        loaders.append(load_data(fusar_path + "/train", batch_size=batch_size))
        loaders.append(load_data(fusar_path + "/val", batch_size=batch_size))
        loaders.append(load_data(fusar_path + "/test", batch_size=batch_size))
    
    return loaders

def calculate_centroids(features, labels, num_classes):
    centroids = []
    for c in range(num_classes):
        class_features = features[labels == c]
        centroids.append(class_features.mean(dim=0))
    return centroids

def generate_synthetic_samples(features, labels, centroids, target_class, alpha=0.1):
    synthetic_features = []
    synthetic_labels = []
    
    for feature, label in zip(features, labels):
        if label != target_class:  # Only translate from majority classes
            translated_feature = feature + alpha * (centroids[target_class] - feature)
            synthetic_features.append(translated_feature)
            synthetic_labels.append(target_class)

    return torch.stack(synthetic_features), torch.tensor(synthetic_labels)

def m2m_creation(train_loader, feature_extractor, classes):
    # Extract features and labels
    features = []
    labels = []
    synthetic_features = []
    synthetic_labels = []

    for inputs, targets in train_loader:
        with torch.no_grad():
            inputs = inputs.cuda()  # Move to GPU if available
            extracted_features = feature_extractor(inputs).mean(dim=(2, 3))  # Global average pooling
            features.append(extracted_features)
            labels.append(targets)
    features = torch.cat(features, dim=0)  # Combine all feature tensors
    labels = torch.cat(labels, dim=0)      # Combine all label tensors

    num_classes = len(classes)
    centroids = calculate_centroids(features, labels, num_classes)

    # Identify minority classes (e.g., classes with fewer than a threshold number of samples)
    class_counts = torch.bincount(labels)
    minority_classes = torch.where(class_counts < 100)[0]  # Adjust threshold as needed

    for target_class in minority_classes:
        features_aug, labels_aug = generate_synthetic_samples(features, labels, centroids, target_class)
        synthetic_features.append(features_aug)
        synthetic_labels.append(labels_aug)

    synthetic_features = torch.cat(synthetic_features, dim=0)
    synthetic_labels = torch.cat(synthetic_labels, dim=0)

    # Combine original and synthetic data
    augmented_features = torch.cat([features, synthetic_features], dim=0)
    augmented_labels = torch.cat([labels, synthetic_labels], dim=0)

    input_size = augmented_features.size(1)  # Size of feature vector

    # Create DataLoader
    augmented_dataset = TensorDataset(augmented_features, augmented_labels)
    augmented_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
    # input_size = augmented_features.size(1)
    
    return input_size, augmented_loader

def train_save(c_datasets_b, models, l_rate, device, epochs, classes, lf, ds):
    csv_res = []
    csv1 = []
    ds = ""
    for dataset_name, dataset_loader in c_datasets_b.items():
        print("Training on ", dataset_name)
        ds = dataset_name
        results += "Training on " + dataset_name + "\n"
        # mix_path = "mix_5"
        train_loader_m = dataset_loader[0]
        val_loader_m = dataset_loader[1]
        test_loader_m = dataset_loader[2]
        for model_name, model in models.items():
            ip, dataset_loader = m2m_creation(train_loader=dataset_loader, feature_extractor=model.features, classes=classes)
            print("Training using :" , model_name)
            results += "Training using "+model_name + "\n"
                
            n_model = MinorityClassClassifier(input_size=ip, classes=classes)
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
                print("Validation: ", dataset_name)
                results += f"Validation {model_name} on {dataset_name} \n"
                str_results, _ = evaluate_model(n_model, val_loader_m, device=device)
                results += str_results
            print("Testing: ", dataset_name)
            results += f"Testing {model_name} on {dataset_name} \n"
            str_results, csv_scores = evaluate_model(n_model, test_loader_m, device=device)
            results += str_results
            csv_res.append(csv_scores)
            csv1.append(csv_scores)
            
        # save_model(n_model, save_dir, model_filename)

    # Open the file in write mode ("w") and write the string to it
    with open(f"scores/sampling_results_{ds}.txt", "w") as f:
        f.write(results)

    fields = ["Accuracy", "Precision", "Recall", "F1"] * (classes +1)

    with open(f"scores/sampling_results_{ds}.csv", 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(csv1)