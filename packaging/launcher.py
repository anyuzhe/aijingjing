from multiprocessing import freeze_support


if __name__ == "__main__":
    freeze_support()

    # Keep expensive GUI imports after PyInstaller's multiprocessing dispatch.
    # A spawned transcription worker can then enter its target without first
    # booting Qt and the entire desktop application.
    from media_knowledge.desktop.app import main

    main()
