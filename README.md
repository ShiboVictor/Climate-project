# Climate-project
Prediction of Extreme Weather Events: Machine Learning Models Trained on Multivariate Meteorological Data

WANG SHIBO

Faculty of Computer Science

shibowang43@gmail.com

The increasing frequency and intensity of extreme weather events due to global warming poses severe threats to life and property across Asia. Traditional statistical methods struggle to capture the complex temporal and nonlinear characteristics of meteorological data, limiting their effectiveness in predicting extreme events. 
This study addresses these limitations by proposing hybrid CEEMDAN-LSTM and CEEMDAN-CNN-LSTM models for predicting droughts and floods using multivariate meteorological data. The models employ Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) to decompose climate signals into multiple time-scale components, combined with Convolutional Neural Networks (CNN) for feature extraction and Long Short-Term Memory (LSTM) networks for temporal prediction. Using monthly meteorological data from eight Asian countries spanning 1950-2023, including 15 variables such as precipitation, consecutive dry days, and temperature extremes, the models follow a "decomposition-reconstruction-prediction-integration" framework. Results demonstrate that CEEMDAN-based models generally outperform standard LSTM, particularly for drought prediction where MAPE improved from 7.78% to 7.20% in China. Model performance varied by region and climate type, with stronger results in monsoon-dominated areas compared to tropical maritime regions. The findings provide valuable insights for developing early warning systems and climate adaptation policies, supporting UN Sustainable Development Goal 13 (Climate Action) through enhanced extreme weather prediction capabilities.
Keywords:  Machine Learning, CEEMDAN, CNN, LSTM, Time-series
