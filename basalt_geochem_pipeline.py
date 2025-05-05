if use_zip:
    import requests
    from io import StringIO

    # Point at the nine JETOA basalt parts in your main branch
    github_csv_urls = [
        f"https://raw.githubusercontent.com/holderds/BasaltChem/main/2024-12-2JETOA_BASALT_part{i}.csv"
        for i in range(1, 10)
    ]

    df_list = []
    skipped_urls = []
    for url in github_csv_urls:
        try:
            # force a consistent dtype read
            resp = requests.get(url)
            resp.raise_for_status()
            df_part = pd.read_csv(StringIO(resp.text), low_memory=False)

            # standardize column names & types
            df_part.rename(columns=rename_map, inplace=True)   # reuse your rename_map from load_data()
            for c in df_part.columns:
                # attempt numeric conversion where possible
                df_part[c] = pd.to_numeric(df_part[c], errors="ignore")

            # drop parts missing essential SiO2 column
            if 'SiO2' not in df_part.columns:
                skipped_urls.append(url)
                continue

            df_list.append(df_part)

        except Exception:
            skipped_urls.append(url)

    if not df_list:
        st.error("❌ Could not load any remote basalt parts from GitHub.")
        st.stop()

    # concatenate, filter, compute ratios
    df = pd.concat(df_list, ignore_index=True)
    df = filter_basalt_to_basaltic_andesite(df)
    df = compute_ratios(df)

    if skipped_urls:
        st.warning(f"⚠ Skipped {len(skipped_urls)} remote file(s):\n• " + "\n• ".join(skipped_urls))

