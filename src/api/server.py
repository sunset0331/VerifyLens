"""
api/server.py
-------------
FastAPI application for VerifyLens.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io

from src.api.models import VerificationResponse
from src.api.pipeline import pipeline
from src.utils.image_utils import load_image

app = FastAPI(
    title="VerifyLens API",
    description="Multimodal KYC Identity Verification",
    version="1.0.0",
)


@app.post("/verify", response_model=VerificationResponse)
async def verify_identity(
    id_image: UploadFile = File(..., description="Image of the ID document"),
    selfie_image: UploadFile = File(..., description="Selfie image of the user"),
):
    """
    Submit an ID document and a selfie to verify identity.
    """
    try:
        id_bytes = await id_image.read()
        selfie_bytes = await selfie_image.read()

        # Load images
        try:
            id_img = load_image(id_bytes)
            selfie_img = load_image(selfie_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # Run pipeline
        result = await pipeline.run(id_img, selfie_img)
        
        return VerificationResponse(**result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=VerificationResponse(
                status="error",
                verdict="REJECTED",
                confidence=0.0,
                error_message=str(e)
            ).model_dump()
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
