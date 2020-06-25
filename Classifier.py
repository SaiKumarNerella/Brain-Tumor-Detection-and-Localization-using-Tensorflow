Import Detection_Utils

ex_img = cv2.imread('yes/Y1.jpg')
ex_new_img = crop_brain_contour(ex_img, True)

augmented_path = 'augmented data/'

# augmented data (yes and no) contains both the original and the new generated examples
augmented_yes = augmented_path + 'yes'
augmented_no = augmented_path + 'no'

IMG_WIDTH, IMG_HEIGHT = (240, 240)

X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))

plot_sample_images(X, y)

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

print("number of training examples = " + str(X_train.shape[0]))
print("number of development examples = " + str(X_val.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(y_train.shape))
print("X_val (dev) shape: " + str(X_val.shape))
print("Y_val (dev) shape: " + str(y_val.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(y_test.shape))

IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

model = build_model(IMG_SHAPE)

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')

filepath = "cnn-parameters-improvement"
# save the model with the best validation (development) accuracy till now
checkpoint = ModelCheckpoint(
    "models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))

start_time = time.time()

model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val),
          callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")

start_time = time.time()

model.fit(x=X_train, y=y_train, batch_size=32, epochs=5, validation_data=(X_val, y_val),
          callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")

start_time = time.time()

model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val),
          callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")

history = model.history.history

for key in history.keys():
    print(key)

plot_metrics(history)

best_model = load_model(filepath='models/cnn-parameters-improvement-23-0.91.model')

loss, acc = best_model.evaluate(x=X_test, y=y_test)

print(f"Test Loss = {loss}")
print(f"Test Accuracy = {acc}")

y_test_prob = best_model.predict(X_test)

f1score = compute_f1_score(y_test, y_test_prob)
print(f"F1 score: {f1score}")

y_val_prob = best_model.predict(X_val)

f1score_val = compute_f1_score(y_val, y_val_prob)
print(f"F1 score: {f1score_val}")

data_percentage(y)

print("Training Data:")
data_percentage(y_train)
print("Validation Data:")
data_percentage(y_val)
print("Testing Data:")
data_percentage(y_test)