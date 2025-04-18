import os
import yaml
import glob

def fix_mlflow_paths():
    """
    Updates all MLflow meta.yaml files to use the current project directory.
    Run this script whenever you move the project to a new location.
    """
    # Get the absolute path to the current project directory
    project_dir = os.path.abspath(os.path.dirname(__file__))
    mlruns_dir = os.path.join(project_dir, 'mlruns')
    
    if not os.path.exists(mlruns_dir):
        print(f"MLflow directory not found at: {mlruns_dir}")
        return
    
    # Find all meta.yaml files in the mlruns directory structure
    meta_files = glob.glob(os.path.join(mlruns_dir, '**', 'meta.yaml'), recursive=True)
    
    updated_count = 0
    for meta_file in meta_files:
        try:
            with open(meta_file, 'r') as f:
                content = yaml.safe_load(f)
            
            # Check if the file has an artifact_location field with an absolute path
            if 'artifact_location' in content and 'file:' in content['artifact_location']:
                # Extract the relative path from the full path
                old_path = content['artifact_location'].split('file:')[1]
                relative_path = old_path[old_path.find('mlruns'):]
                
                # Create the new path based on the current project directory
                new_path = f"file:{os.path.join(project_dir, relative_path)}"
                
                # Only update if the path has actually changed
                if content['artifact_location'] != new_path:
                    content['artifact_location'] = new_path
                    
                    # Write the updated content back to the file
                    with open(meta_file, 'w') as f:
                        yaml.dump(content, f, default_flow_style=False)
                    
                    updated_count += 1
                    print(f"Updated: {meta_file}")
        
        except Exception as e:
            print(f"Error processing {meta_file}: {str(e)}")
    
    print(f"\nCompleted! Updated {updated_count} MLflow meta.yaml files.")

if __name__ == "__main__":
    fix_mlflow_paths()