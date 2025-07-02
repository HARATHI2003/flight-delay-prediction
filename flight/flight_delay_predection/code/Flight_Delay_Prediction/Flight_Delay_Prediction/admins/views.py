from django.shortcuts import render,redirect
from django.contrib import messages
from users.models import Reg_User
# Create your views here.
def adminbase(request):
    return render(request,'admins/adminbase.html')

def viewuser(request):
    register = Reg_User.objects.all()
    return render(request,'admins/viewusers.html',{'register':register})

def register(request):
    if request.method == 'POST':
        name = request.POST.get('username')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        address = request.POST.get('Address')
        password = request.POST.get('psw')
        status = request.POST.get('status')
        try:
          if name and email and phone and address and password and status:
            register = Reg_User(username=name,email=email,phone_number=phone,address=address,password=password,status=status)
            register.save()
            messages.success(request,'User Registered Successfully')
            print('data saved sucessfully')
            return redirect('register')
          else:
              print('error at register')
              messages.success(request,'User not Registered Successfully')
        except Exception as e:
              print(f'the exception is {e}')
              messages.success(request,f'error at register is {str(e)}')      
    return render(request,'user_register.html')

from django.contrib import messages
from django.shortcuts import redirect, render

def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('uname')
        password = request.POST.get('psw')
        print(f"Username: {username}, Password: {password}")
        if username == 'admin' and password == 'admin':            
            return redirect('adminbase')
        else:
            messages.error(request, 'Invalid Credentials')
    print('admin login page rendering')
    return render(request, 'adminlogin.html')

    
    