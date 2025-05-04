import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)
from utils.WRN import *

device = torch.device('cuda')

if __name__ == "__main__":
    model = WRN(40, 2, 10, 0.25)

    stateDict = torch.load("./FixMatch/results/Resnet_40_2.pt")
    model.load_state_dict(stateDict)
    model = model.to(device)
    
    test = torch.load("./datasets/stl-10/test.pt"); testX = test[0].to(torch.float32).permute(0, 3, 1, 2); testY = test[1].long(); del test
    testX /= 255
    testDS = TensorImageDataset(testX, testY)
    testLoader = DataLoader(testDS, 500, shuffle = True)
    
    model.eval()
    valCount = 0
    with torch.no_grad():
        for xBatch, yBatch in testLoader:

            logits       = model(xBatch.to(device))
            distribution = torch.softmax(logits, dim = 1).to(torch.device('cpu'))

            valCount     += (torch.argmax(distribution, dim = 1) == yBatch).sum().item()
            
    print(f"Validation Accuracy: {(valCount / testY.numel()) * 100:.2f}%")