from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [

    path('', views.dashboard, name='dashboard'),
    path('login/', views.login_view , name='login'),
    path('register/', views.register_view , name='register'),
    path('logout/', views.logout_view , name='logout'),

    path('analytics/', views.analytics , name='analytics'),
    path('predictions/', views.predictions),
    path('regions/', views.regions , name='regions'),
    path('feedback/', views.feedback),
    

]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)