#!/usr/bin/env python3
"""
Netflix Recommendation System - Project Structure Display

This script shows the complete project structure and provides
an overview of all components, files, and their purposes.
"""

import os
from pathlib import Path
from datetime import datetime

def get_file_info(filepath):
    """Get file size and modification date"""
    try:
        stat = filepath.stat()
        size = stat.st_size
        modified = datetime.fromtimestamp(stat.st_mtime)
        
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024**2:
            size_str = f"{size/1024:.1f} KB"
        elif size < 1024**3:
            size_str = f"{size/(1024**2):.1f} MB"
        else:
            size_str = f"{size/(1024**3):.1f} GB"
            
        return size_str, modified.strftime("%Y-%m-%d %H:%M")
    except:
        return "N/A", "N/A"

def display_tree(path, prefix="", max_depth=3, current_depth=0):
    """Display directory tree structure"""
    if current_depth > max_depth:
        return
        
    path = Path(path)
    if not path.exists():
        return
        
    # Get items and sort (directories first, then files)
    items = []
    try:
        for item in path.iterdir():
            if not item.name.startswith('.') and item.name not in ['__pycache__', '.venv']:
                items.append(item)
    except PermissionError:
        return
        
    # Sort: directories first, then files
    items.sort(key=lambda x: (x.is_file(), x.name.lower()))
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        
        if item.is_dir():
            print(f"{prefix}{current_prefix}{item.name}/")
            extension = "    " if is_last else "│   "
            display_tree(item, prefix + extension, max_depth, current_depth + 1)
        else:
            size, modified = get_file_info(item)
            print(f"{prefix}{current_prefix}{item.name} ({size})")

def show_project_overview():
    """Show complete project overview"""
    
    print("🎬 NETFLIX RECOMMENDATION SYSTEM - PROJECT STRUCTURE")
    print("=" * 70)
    
    project_root = Path.cwd()
    
    print(f"\n📁 Project Root: {project_root}")
    print(f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 70)
    
    # Display directory tree
    print("\n📂 PROJECT STRUCTURE:")
    print("-" * 30)
    display_tree(project_root, max_depth=3)
    
    # Show key components
    print(f"\n📋 KEY COMPONENTS OVERVIEW:")
    print("=" * 50)
    
    components = [
        {
            'folder': 'data/',
            'description': 'Data Storage and Management',
            'subfolders': [
                ('raw/', 'Original Netflix movies dataset'),
                ('cleaned/', 'Preprocessed and cleaned data'),
                ('final/', 'Feature-engineered data ready for ML'),
                ('processed/', 'Intermediate processing results')
            ]
        },
        {
            'folder': 'models/',
            'description': 'Trained Models and Artifacts',
            'subfolders': [
                ('trained/', 'Final trained models (.joblib, .pkl)'),
                ('checkpoints/', 'Training checkpoints and intermediate models')
            ]
        },
        {
            'folder': 'results/',
            'description': 'Analysis Results and Reports',
            'subfolders': [
                ('images/', 'Visualization plots and charts'),
                ('metrics/', 'Performance metrics and feature importance'),
                ('reports/', 'Summary reports and project documentation')
            ]
        },
        {
            'folder': 'src/',
            'description': 'Source Code and Utilities',
            'files': [
                ('netflix_recommendation_system.ipynb', 'Main Jupyter notebook'),
                ('model_loader.py', 'Model loading and display utilities'),
                ('data_utils.py', 'Data processing and feature engineering'),
                ('model_utils.py', 'Model training and evaluation utilities'),
                ('train_models.py', 'Complete training pipeline script')
            ]
        }
    ]
    
    for comp in components:
        print(f"\n📁 {comp['folder']}")
        print(f"   Purpose: {comp['description']}")
        
        if 'subfolders' in comp:
            for subfolder, desc in comp['subfolders']:
                print(f"   ├── {subfolder} - {desc}")
                
        if 'files' in comp:
            for filename, desc in comp['files']:
                print(f"   ├── {filename} - {desc}")
    
    # Show file counts and sizes
    print(f"\n📊 PROJECT STATISTICS:")
    print("=" * 30)
    
    def count_files_in_dir(directory, extensions=None):
        """Count files in directory with optional extension filter"""
        if not Path(directory).exists():
            return 0, 0
            
        count = 0
        total_size = 0
        
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file():
                if extensions is None or file_path.suffix.lower() in extensions:
                    count += 1
                    try:
                        total_size += file_path.stat().st_size
                    except:
                        pass
        
        return count, total_size
    
    def format_size(size_bytes):
        """Format file size"""
        if size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
    
    stats = [
        ('Data Files', 'data/', ['.csv', '.json', '.parquet']),
        ('Model Files', 'models/', ['.joblib', '.pkl']),
        ('Result Files', 'results/', ['.csv', '.json', '.png']),
        ('Source Code', 'src/', ['.py', '.ipynb']),
        ('Documentation', '.', ['.md', '.txt'])
    ]
    
    for name, directory, extensions in stats:
        count, size = count_files_in_dir(directory, extensions)
        print(f"{name:15}: {count:3} files ({format_size(size)})")
    
    # Show important files
    print(f"\n🎯 KEY FILES:")
    print("=" * 20)
    
    key_files = [
        'netflix_movies_detailed_up_to_2025.csv',
        'src/netflix_recommendation_system.ipynb', 
        'src/model_loader.py',
        'models/trained/netflix_recommendation_model_final.joblib',
        'results/metrics/model_comparison.csv',
        'README.md'
    ]
    
    for file in key_files:
        file_path = Path(file)
        if file_path.exists():
            size, modified = get_file_info(file_path)
            print(f"✅ {file} ({size}) - Modified: {modified}")
        else:
            print(f"❌ {file} - Not found")
    
    print(f"\n🚀 PROJECT STATUS:")
    print("=" * 20)
    
    # Check project completeness
    required_components = [
        ('Data', 'data/raw/netflix_movies_detailed_up_to_2025.csv'),
        ('Models', 'models/trained/'),
        ('Results', 'results/metrics/'),
        ('Source Code', 'src/model_loader.py')
    ]
    
    all_complete = True
    for name, path in required_components:
        if Path(path).exists():
            print(f"✅ {name}: Available")
        else:
            print(f"❌ {name}: Missing")
            all_complete = False
    
    if all_complete:
        print(f"\n🎉 PROJECT IS COMPLETE AND READY!")
        print(f"   Run: python src/model_loader.py")
        print(f"   Or:  python src/train_models.py")
    else:
        print(f"\n⚠️  Some components are missing. Please complete setup.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    show_project_overview()