from django.db import models

class AudioFile(models.Model):
    """
    Model to store the uploaded audio files and their diagnosis.
    """
    file = models.FileField(upload_to='uploads/')  # Store audio files
    diagnosis = models.CharField(max_length=255, blank=True, null=True)  # Store diagnosis (COPD, Healthy, etc.)
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Timestamp for file upload

    def __str__(self):
        return f"{self.file.name} - {self.diagnosis}"
