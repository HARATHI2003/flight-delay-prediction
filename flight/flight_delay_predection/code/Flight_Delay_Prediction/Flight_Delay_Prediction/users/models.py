from django.db import models

# Create your models here.
from django.db import models

class Reg_User(models.Model):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    phone_number = models.CharField(max_length=15)
    password = models.CharField(max_length=128)  # Ideally, you would hash the password
    address = models.TextField()
    status = models.CharField(default='waiting', max_length=10)

    def __str__(self):
        return self.username
