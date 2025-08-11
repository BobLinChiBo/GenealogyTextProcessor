@echo off
echo Creating .env file from .env.example...

if exist .env (
    echo .env file already exists!
    set /p overwrite="Do you want to overwrite it? (y/n): "
    if /i not "%overwrite%"=="y" (
        echo Cancelled.
        exit /b
    )
)

if exist .env.example (
    copy .env.example .env
    echo .env file created successfully!
    echo.
    echo Now edit .env and replace 'your-api-key-here' with your actual Gemini API key.
    echo You can get an API key from: https://makersuite.google.com/app/apikey
    echo.
    notepad .env
) else (
    echo .env.example file not found!
    echo Creating a new .env file...
    (
        echo # Gemini API Configuration
        echo GEMINI_API_KEY=your-api-key-here
    ) > .env
    echo .env file created!
    notepad .env
)