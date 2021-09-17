import torchvision
import numpy
import torch
import segmentation_utils
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import glob
from datetime import datetime


# set computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

images = []
for img in glob.glob("train_images/*.png")[:300]:
    images.append(transform(Image.open(img)).to(device))
labels = []
for img in glob.glob("train_labels/*.png")[:300]:
    labels.append(torch.Tensor(numpy.array(Image.open(img))).long().to(device))


class ImgDataSet(Dataset):

    def __init__(self):
        length = len(images)
        self.len = int(length * 0.9)

        # Concatenate the two data sets
        self.images = images[:self.len]
        self.labels = labels[:self.len]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.len


class TestDataSet(Dataset):

    def __init__(self):
        length = len(images)
        ind = int(length * 0.9)
        self.len = length - ind

        # Concatenate the two data sets
        self.images = images[ind:]
        self.labels = labels[ind:]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.len


img_set = ImgDataSet()
tst_set = TestDataSet()
loader = DataLoader(dataset=img_set, batch_size=1, shuffle=True, )
test_loader = DataLoader(dataset=tst_set, batch_size=1, shuffle=False, )

# download or load the model from disk
model = torchvision.models.segmentation.fcn_resnet101(num_classes=9).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

all_iou_train = []
all_iou_test = []

now = datetime.now()
print('Start time: ', now)

max_test_iou = 0

for i in range(100):
    train_iou = []
    total_loss = []

    for j, data in enumerate(loader):
        # do forward pass and get the output dictionary
        image = data[0]
        label_ts = data[1]
        outputs = model(image)
        # get the data from the `out` key
        outputs = outputs['out']
        result = torch.movedim(outputs, 1, 3)
        new_shape = (outputs.shape[0] * outputs.shape[2] * outputs.shape[3], outputs.shape[1])
        result = result.reshape(new_shape)
        target = label_ts.reshape((-1))

        optimiser.zero_grad()
        loss = criterion(result, target)

        for n in range(len(outputs)):
            train_iou.append(segmentation_utils.calculate_IoU(outputs[n], label_ts[n]))

        loss.backward()
        optimiser.step()
        total_loss.append(loss.item())
    print()
    print('Epoch:', i, 'Train Loss:', sum(total_loss)/len(total_loss), 'Train IoU:', numpy.mean(train_iou).item())
    all_iou_train.append(numpy.mean(train_iou).item())

    test_iou = []
    for k, data in enumerate(test_loader):
        # do forward pass and get the output dictionary
        image = data[0]
        label_ts = data[1]
        outputs = model(image)
        # get the data from the `out` key
        outputs = outputs['out']

        for n in range(len(outputs)):
            test_iou.append(segmentation_utils.calculate_IoU(outputs[n], label_ts[n]))

    print('Epoch:', i, 'Test IoU:', numpy.mean(test_iou).item())
    all_iou_test.append(numpy.mean(test_iou).item())
    if numpy.mean(test_iou).item() > max_test_iou:
        torch.save(model.state_dict(), 'saved_model.pt')


then = datetime.now()
print('Time elapsed: ', then - now)


plt.plot(range(len(all_iou_train)), all_iou_train, color='red', marker='o')
plt.plot(range(len(all_iou_test)), all_iou_test, color='blue', marker='o')
plt.show()
