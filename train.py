import torch
from torchvision import datasets, transforms
import os
import pretrainedmodels
from util import path, train_path
import split_data_train_test

data_dir = path

if not os.path.exists(train_path):
    split_data_train_test.generate()

data_transform = {x: transforms.Compose([transforms.ColorJitter(brightness=0.2),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                  for x in ["train", "test"]}

image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                          transform=data_transform[x])
                  for x in ["train", "test"]}
dataloader = {x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                             batch_size=64,
                                             shuffle=True)
              for x in ["train", "test"]}


model = pretrainedmodels.se_resnext50_32x4d()
model.last_linear = torch.nn.Linear(in_features=2048, out_features=3, bias=True)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # weight_decay=0.0001
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7], gamma=0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
best_acc = 0.0
model_name = "se_resnext50_32x4d"
save_path = './{}.pth'.format(model_name)

for epoch in range(50):
    model.train()
    running_loss = 0.0
    for step, (img, label) in enumerate(dataloader["train"]):
        optimizer.zero_grad()
        outputs = model(img.to(device))
        loss = loss_function(outputs, label.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(dataloader["train"])
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    lr = lr_scheduler.get_lr()[0]
    lr_scheduler.step()

    # validate
    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for data_test in dataloader["test"]:
            test_images, test_labels = data_test
            optimizer.zero_grad()
            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]

            acc += (predict_y == test_labels.to(device)).sum().item()

        accurate_test = round(acc / len(image_datasets["test"]), 4)

        if accurate_test >= best_acc:
            best_acc = accurate_test
            torch.save(model.state_dict(), save_path)
            print("save weights")

        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f  lr:%.7f' %
              (epoch + 1, running_loss / step, accurate_test, lr))

print('Finished Training')
