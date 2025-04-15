2025-04-15 19:45:35.402610: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-04-15 19:45:37.487608: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Successfully loaded sheet 1 with refractive index 1.0
Successfully loaded sheet 2 with refractive index 1.2
Successfully loaded sheet 3 with refractive index 1.3
Successfully loaded sheet 4 with refractive index 1.33
Successfully loaded sheet 5 with refractive index 1.4
Successfully loaded sheet 6 with refractive index 1.5
Available columns: ['diameter', 'refractive_index', 'resonance_wavelength', 'extinction']
Loaded 1800 data points
Successfully loaded existing models
Traceback (most recent call last):
  File "C:\Users\AM323\OneDrive\Desktop\Nanomaterial_Project\code1\newmodel\NEW2\forward3.py", line 757, in <module>
    grid_results = model.generate_prediction_grid()
  File "C:\Users\AM323\OneDrive\Desktop\Nanomaterial_Project\code1\newmodel\NEW2\forward3.py", line 664, in generate_prediction_grid
    wavelength, extinction = self.predict(d, ri)
                             ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\AM323\OneDrive\Desktop\Nanomaterial_Project\code1\newmodel\NEW2\forward3.py", line 511, in predict
    model = self.models[closest_ri]
            ~~~~~~~~~~~^^^^^^^^^^^^
KeyError: 1.0






Epoch 215/500
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - extinction_loss: 7.7751e-04 - extinction_mae: 0.0208 - loss: 0.0014 - wavelength_loss: 6.0643e-04 - wavelength_mae: 0.0183 - val_extinction_loss: 8.8456e-05 - val_extinction_mae: 0.0072 - val_loss: 2.7436e-04 - val_wavelength_loss: 1.8590e-04 - val_wavelength_mae: 0.0088 - learning_rate: 1.5625e-05
Epoch 216/500
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - extinction_loss: 0.0012 - extinction_mae: 0.0277 - loss: 0.0021 - wavelength_loss: 8.8335e-04 - wavelength_mae: 0.0220 - val_extinction_loss: 9.0711e-05 - val_extinction_mae: 0.0074 - val_loss: 2.7204e-04 - val_wavelength_loss: 1.8133e-04 - val_wavelength_mae: 0.0086 - learning_rate: 1.5625e-05
Epoch 217/500
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - extinction_loss: 7.1613e-04 - extinction_mae: 0.0198 - loss: 0.0014 - wavelength_loss: 6.3730e-04 - wavelength_mae: 0.0185 - val_extinction_loss: 8.7691e-05 - val_extinction_mae: 0.0071 - val_loss: 2.5898e-04 - val_wavelength_loss: 1.7129e-04 - val_wavelength_mae: 0.0083 - learning_rate: 1.5625e-05
Epoch 218/500
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - extinction_loss: 9.0560e-04 - extinction_mae: 0.0220 - loss: 0.0017 - wavelength_loss: 8.3623e-04 - wavelength_mae: 0.0204 - val_extinction_loss: 8.2507e-05 - val_extinction_mae: 0.0070 - val_loss: 2.5672e-04 - val_wavelength_loss: 1.7421e-04 - val_wavelength_mae: 0.0084 - learning_rate: 1.5625e-05
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 108ms/step
RI = 1.0 - RMSE Wavelength: 4.7062, R² Wavelength: 0.9958
RI = 1.0 - RMSE Extinction: 3182.026598, R² Extinction: 0.9981
Traceback (most recent call last):
  File "C:\Users\AM323\OneDrive\Desktop\Nanomaterial_Project\code1\newmodel\NEW2\forward2.py", line 937, in main
    model.train_model(combined_df)
  File "C:\Users\AM323\OneDrive\Desktop\Nanomaterial_Project\code1\newmodel\NEW2\forward2.py", line 286, in train_model
    self.plot_predictions(data, y_wavelength_pred, y_extinction_pred, ri)
  File "C:\Users\AM323\OneDrive\Desktop\Nanomaterial_Project\code1\newmodel\NEW2\forward2.py", line 459, in plot_predictions
    model = self.models[ri] if ri is not None else self.models['unified']
            ~~~~~~~~~~~^^^^
KeyError: 1.0
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "C:\Users\AM323\OneDrive\Desktop\Nanomaterial_Project\code1\newmodel\NEW2\forward2.py", line 982, in <module>
    main()
  File "C:\Users\AM323\OneDrive\Desktop\Nanomaterial_Project\code1\newmodel\NEW2\forward2.py", line 941, in main
    model.train_model(combined_df)
  File "C:\Users\AM323\OneDrive\Desktop\Nanomaterial_Project\code1\newmodel\NEW2\forward2.py", line 286, in train_model
    self.plot_predictions(data, y_wavelength_pred, y_extinction_pred, ri)
  File "C:\Users\AM323\OneDrive\Desktop\Nanomaterial_Project\code1\newmodel\NEW2\forward2.py", line 459, in plot_predictions
    model = self.models[ri] if ri is not None else self.models['unified']
            ~~~~~~~~~~~^^^^
KeyError: 1.0





