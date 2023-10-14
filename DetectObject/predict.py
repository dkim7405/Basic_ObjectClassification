import torch
from torchvision import transforms
from models.my_model import MyModel

def prediction(model, batch_input):
    model.eval()

    data = batch_input.to(model.device)

    output = model(data)

    prob = torch.nn.functional.softmax(output, dim=1)

    pred_prob, pred_index = prob.max(1)

    return pred_index.cpu().numpy(), pred_prob.cpu().numpy()

if __name__ == "__main":
    # Load your trained model
    model = MyModel()
    model.load_state_dict(torch.load('models/cifar10_cnn_model.pt'))

    # Define the transformation for test data (make sure it matches the preprocessing used during training)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Example values, adapt to your training data
    ])

    # Load an example batch of test data
    # Replace this with your actual test data loading logic
    example_data = torch.rand(5, 3, 32, 32)  # Replace with your test data

    # Apply the test data transformation
    example_data = test_transform(example_data)

    # Perform prediction
    predictions, probabilities = prediction(model, example_data)

    # Display predictions
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for i in range(len(predictions)):
        print(f"Prediction: {classes[predictions[i]]}, Probability: {probabilities[i]}")
