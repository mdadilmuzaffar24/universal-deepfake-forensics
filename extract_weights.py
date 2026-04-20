import tensorflow as tf

print("Loading full model...")
# Point it to your local models folder
model = tf.keras.models.load_model('./models/xception_MASTER_rehearsal.keras')

print("Exporting weights only...")
# Save the pure mathematical weights
model.save_weights('./models/xception_weights_only.weights.h5')

print("✅ Weights extracted successfully! Check your models folder.")