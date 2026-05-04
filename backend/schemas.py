from pydantic import BaseModel

class UploadResponse(BaseModel):
    message: str
    total_inserted: int
    total_scored: int
