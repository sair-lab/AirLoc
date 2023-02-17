### Data Prerpocessing

Expected directory structure for preprocessing is:

```jsx
Base Directory/
└──  dataset(mp3d)/
			└── scene(RPmz2sHmrrY)/
					└── habitat
					└── rooms
							└── room(bedroom)
									└── raw_data
									└── viz_data
									└── poses_cleaned.json

```

1. Superpoint extraction (Modify superpoint_extraction_airloc.yaml)

        

```jsx
python superpoint_extraction_airloc.py -c config/superpoint_extraction_airloc.yaml
```

1. Preprocessing does the following
    1. Removes bad image (Having very less number of object, black outs and all) 
    2. Makes a .pkl file per dataset into specified dataset directory
    
    ```jsx
    python datasets/preprocess_reloc.py -c config/preprocess_reloc.yaml
    ```
