"""
URL configuration for template2 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from users.views import *
from admins.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',index,name='index'),
    path('register/',register,name='register'),

    #user urls
    path('userlogin/',userlogin,name='userlogin'),  
    path('userbase',userbase,name='userbase'), 
    path("training/",training,name='training'), 
    path("prediction",prediction,name='predict'),    

    #admin urls
    path('adminlogin/',admin_login,name='adminlogin'),   
    path('adminbase',adminbase,name='adminbase'),  
    path('viewusers',viewuser,name='viewusers'),   
    path('activate<int:id>',ActivateUser,name='activate'),     
    path('blockuser<int:id>',BlockUser,name='blockuser'),                                            
]
