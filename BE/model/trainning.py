from model import EyeStateModel

model = EyeStateModel()

model.train(train_dir = "data/train", val_dir = "data/test",epochs = 15)
model.save("eye_model.h5")
