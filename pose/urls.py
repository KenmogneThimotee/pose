from django.urls import path
from .views import comparePose, setImageSource

urlpatterns = [
    path("comparePose/", comparePose, name="compare"),
    path("setsource/", setImageSource, name="set")
]

