from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a trained model on the test set.

    Parameters:
    model: The trained machine learning model to evaluate.
    X_test: The feature data for the test set.
    y_test: The true labels for the test set.

    Returns:
    dict: A dictionary containing evaluation metrics such as accuracy, classification report, and confusion matrix.
    """
    
    print("Evaluating model performance...")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(class_report)
    print("Confusion Matrix:")
    print(conf_matrix)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix
    }