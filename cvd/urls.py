
from django.urls import path
from . import views
urlpatterns = [
    path("", views.home, name="home"),
    path("preliminary-entry/", views.preliminary_entry, name="preliminary-entry"),
  
    
    path("computation/", views.computation, name="computation"),
    
    path("greater-than-35/", views.greater_than_35, name="greater-than"),
    path("less-than-35/", views.less_than_35, name="less-than-35"),
    
    path("result/", views.result, name="result"),
    path("full-results/", views.full_results, name="full-res"),
    
   # path("print/", views.print, name="print"),
    
    path("sample/", views.sample, name="sample")
    
]
