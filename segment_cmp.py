import torchvision
import numpy
import torch
import glob
import segmentation_utils
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

# set computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create a transform function for data normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# load the images and labels to the computation device
# it's noted that the images should have same dimension with the labels.
images = []
for img in glob.glob("train_images/*.png"):
    images.append(transform(Image.open(img)).to(device))
labels = []
for img in glob.glob("train_labels/*.png"):
    labels.append(torch.Tensor(numpy.array(Image.open(img))).long().to(device))


class ImgDataSet(Dataset):
    # The dataset for training images and labels
    def __init__(self):
        self.len = int(len(images) * 0.9)
        self.images = images[:self.len]
        self.labels = labels[:self.len]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.len


class TestDataSet(Dataset):
    # The dataset for test images and labels
    def __init__(self):
        length = len(images)
        ind = int(length * 0.9)
        self.len = length - ind
        self.images = images[ind:]
        self.labels = labels[ind:]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.len


# define the data sets and loaders
img_set = ImgDataSet()
tst_set = TestDataSet()
train_loader = DataLoader(dataset=img_set, batch_size=1, shuffle=True, )
test_loader = DataLoader(dataset=tst_set, batch_size=1, shuffle=False, )

# load the model from torchvision.models.segmentation
model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(num_classes=9).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# create two lists to record the iou of the model's prediction
all_iou_train = []
all_iou_test = []
max_test_iou = 0

now = datetime.now()
print('Start time: ', now)

for i in range(100):
    train_iou = []
    total_loss = []
    for j, data in enumerate(train_loader):
        optimiser.zero_grad()
        image = data[0]
        label_ts = data[1]
        # do forward pass and get the output dictionary
        outputs = model(image)
        # get the data from the `out` key
        outputs = outputs['out']
        # modify the data dimension from (batch, channel, width, height) to (batch * width * height, class)
        result = torch.movedim(outputs, 1, 3)
        new_shape = (outputs.shape[0] * outputs.shape[2] * outputs.shape[3], outputs.shape[1])
        result = result.reshape(new_shape)
        target = label_ts.reshape((-1))
        loss = criterion(result, target)
        # calculate the IoU of the prediction results
        for n in range(len(outputs)):
            train_iou.append(segmentation_utils.calculate_IoU(outputs[n], label_ts[n]))
        # back-propagate the loss and do optimisation
        loss.backward()
        optimiser.step()
        total_loss.append(loss.item())
    print()
    print('Epoch:', i, 'Train Loss:', sum(total_loss) / len(total_loss), 'Train IoU:', numpy.mean(train_iou).item())
    all_iou_train.append(numpy.mean(train_iou).item())

    # calculate and record the IoU on the test set
    test_iou = []
    for k, data in enumerate(test_loader):
        image = data[0]
        label_ts = data[1]
        # do forward pass and get the output dictionary
        outputs = model(image)
        # get the data from the `out` key
        outputs = outputs['out']

        for n in range(len(outputs)):
            test_iou.append(segmentation_utils.calculate_IoU(outputs[n], label_ts[n]))

    print('Epoch:', i, 'Test IoU:', numpy.mean(test_iou).item())
    all_iou_test.append(numpy.mean(test_iou).item())
    # save the trained model with the highest test IoU
    if numpy.mean(test_iou).item() > max_test_iou:
        torch.save(model.state_dict(), 'saved_model.pt')

then = datetime.now()
print('Time elapsed: ', then - now)

# plot the model's performance
plt.plot(range(len(all_iou_train)), all_iou_train, color='red', marker='o')
plt.plot(range(len(all_iou_test)), all_iou_test, color='blue', marker='o')
plt.show()
