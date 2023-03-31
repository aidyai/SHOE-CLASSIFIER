import uvicorn



if __name__ == "__main__":
  uvicorn.run("src.main:app", host="0.0.0.0", port=10000, reload=True)

#if __name__ == "__main__":
# uvicorn.run("src.main:app", port=8000, reload=True)

