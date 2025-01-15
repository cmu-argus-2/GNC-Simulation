import brahe


def update_brahe_data_files():
    try:
        brahe.utils.download_iers_bulletin_ab(outdir=brahe.constants.DATA_PATH)
        print("Updated IERS Bulletin A")
    except Exception:
        print("Failed to update IERS Bulletin A")
