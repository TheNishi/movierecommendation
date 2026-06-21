import os
import urllib.request
import time

def download_file(url, filename):
    print(f"Starting download of {url} to {filename}...")
    start_time = time.time()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Custom opener to handle headers (like User-Agent) just in case
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    
    try:
        urllib.request.urlretrieve(url, filename)
        duration = time.time() - start_time
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"Successfully downloaded {filename} ({file_size:.2f} MB) in {duration:.1f} seconds.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Let's try an alternative URL if it's the credits file
        if "credits" in url:
            alt_url = "https://raw.githubusercontent.com/manishKrMahto/Movie-Recommender-system/master/tmdb_5000_credits.csv"
            print(f"Attempting fallback to {alt_url}...")
            try:
                urllib.request.urlretrieve(alt_url, filename)
                print(f"Successfully downloaded via fallback.")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise e2
        else:
            raise e

def main():
    movies_url = "https://raw.githubusercontent.com/nirdesh17/movie-recommender-system/master/tmdb_5000_movies.csv"
    credits_url = "https://raw.githubusercontent.com/nirdesh17/movie-recommender-system/master/tmdb_5000_credits.csv"
    
    movies_dest = os.path.join("data", "tmdb_5000_movies.csv")
    credits_dest = os.path.join("data", "tmdb_5000_credits.csv")
    
    download_file(movies_url, movies_dest)
    download_file(credits_url, credits_dest)
    
    print("All dataset downloads completed!")

if __name__ == "__main__":
    main()
