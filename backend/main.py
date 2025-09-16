#!/usr/bin/env python3
"""
AI Meal Planner - FastAPI Application
A plant-based meal planning system with parallel agent execution.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import logging
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import meal planner functionality
from meal_planner import (
    MealPlanRequest, 
    MealPlanResponse, 
    plan_meals
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Meal Planner",
    description="Plant-based meal planning with AI agents",
    version="1.0.0"
)

# Mount static files (frontend)
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.get("/")
async def serve_meal_planner():
    """Serve the meal planner frontend."""
    meal_planner_path = Path(__file__).parent.parent / "frontend" / "meal-planner.html"
    if meal_planner_path.exists():
        return FileResponse(meal_planner_path)
    else:
        return {"message": "Meal Planner frontend not found"}

@app.get("/meal-planner.html")
async def serve_meal_planner_alt():
    """Alternative route for meal planner frontend."""
    return await serve_meal_planner()

@app.post("/plan-meals", response_model=MealPlanResponse)
async def plan_meals_endpoint(request: MealPlanRequest):
    """Main endpoint for meal planning requests."""
    try:
        logger.info(f"Received meal planning request: {request.dietary_preference} for {request.duration}")
        
        result = plan_meals(request)
        
        logger.info(f"Meal plan generated successfully. Result length: {len(result.result)}")
        return result
        
    except Exception as e:
        logger.error(f"Error in meal planning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Meal planning failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "ai-meal-planner",
        "version": "1.0.0"
    }

@app.get("/docs")
async def docs():
    """API documentation endpoint."""
    return {"message": "API documentation available at /docs"}

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting AI Meal Planner server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
