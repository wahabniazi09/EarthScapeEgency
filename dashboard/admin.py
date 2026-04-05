from django.contrib import admin
from .models import *

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role')
    list_filter = ('role',)
    search_fields = ('user__username',)
    

@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('user', 'name', 'rating', 'created_at')
    list_filter = ('rating', 'created_at')
    search_fields = ('name', 'email', 'comments')