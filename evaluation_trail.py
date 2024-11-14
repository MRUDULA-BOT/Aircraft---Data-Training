from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Evaluate the model
def evaluate_model(model, dataloader):
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Don't calculate gradients for evaluation
        for images, labels in dataloader:
            # Move data to GPU if available
            images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get the predicted class

            # Store the true labels and predicted labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataloader.dataset.image_classes, yticklabels=dataloader.dataset.image_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Calculate accuracy
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Call the evaluation function
evaluate_model(model, test_dataloader)
