# Known Faces Directory

Place images of known individuals here using the following structure:

```
known_faces/
├── PersonName1/
│   ├── photo1.jpg
│   └── photo2.jpg
└── PersonName2/
    └── photo1.jpg
```

- Each sub-directory name becomes the person's identity label.
- Supported formats: `.jpg`, `.jpeg`, `.png`
- For best recognition accuracy, use clear frontal-face photos with good lighting.
- Multiple images per person improve matching reliability.

The system computes face embeddings on startup and caches them in `data/models/face_encodings.pkl`.
To re-encode after adding new faces, delete `data/models/face_encodings.pkl` and restart the system.
