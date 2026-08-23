from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('upload/', views.upload_audio, name='upload_audio'),
    # Optionally, add another view for showing the results (handled in the upload_audio view itself)
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
