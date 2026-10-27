"""Se crea este archivo para poder ajustar los parametros no 
se hace directamente en la ablacion ya que se pierde un orden
 pre establecido por el grupo y ademas para no mezclar rutas de 
 trabajo diferentes  """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from ses.config import ROOT
from ses.pipeline import cargar_entrenamiento, construir_pipeline

X, y, cv = cargar_entrenamiento()