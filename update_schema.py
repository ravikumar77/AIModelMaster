from sqlalchemy import create_engine, text

# Create a connection to the database
engine = create_engine('sqlite:///D:/Work/instance/llm_platform.db')

# Execute the ALTER TABLE statement to add the missing columns
with engine.connect() as conn:
    # Check if dataset_id column exists in training_job table
    result = conn.execute(text("PRAGMA table_info(training_job)"))
    columns = [column[1] for column in result.fetchall()]
    
    if 'dataset_id' not in columns:
        print("Adding dataset_id column to training_job table...")
        conn.execute(text('ALTER TABLE training_job ADD COLUMN dataset_id INTEGER REFERENCES coding_dataset(id)'))
        
    if 'training_type' not in columns:
        print("Adding training_type column to training_job table...")
        conn.execute(text('ALTER TABLE training_job ADD COLUMN training_type VARCHAR(64) DEFAULT \'general\''))
    
    # Check if api_key_id column exists in generation_log table
    result = conn.execute(text("PRAGMA table_info(generation_log)"))
    columns = [column[1] for column in result.fetchall()]
    
    if 'api_key_id' not in columns:
        print("Adding api_key_id column to generation_log table...")
        conn.execute(text('ALTER TABLE generation_log ADD COLUMN api_key_id INTEGER REFERENCES api_key(id)'))
    
    conn.commit()
    
    # Verify the columns were added
    print("\nTraining job table columns:")
    result = conn.execute(text("PRAGMA table_info(training_job)"))
    columns = result.fetchall()
    for column in columns:
        print(column)
    
    print("\nGeneration log table columns:")
    result = conn.execute(text("PRAGMA table_info(generation_log)"))
    columns = result.fetchall()
    for column in columns:
        print(column)
        
print('Database schema update completed.')