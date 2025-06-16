from googleapiclient.discovery import build

class YouTubeSearch:
    def __init__(self):
        self.youtube = build("youtube", "v3", developerKey="AIzaSyCXcWwX8DphCUe7LIyW-HPn1uVDNnkJUm8")

    def search_video(self, query):
        try:
            request = self.youtube.search().list(
                q=query,
                part="snippet",
                maxResults=5
            )
            response = request.execute()
            # Extract relevant data from response
            videos = []
            for item in response['items']:
                video = {
                    'title': item['snippet']['title'],
                    'video_url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    'channel_title': item['snippet']['channelTitle'],
                    'publish_time': item['snippet']['publishedAt'],
                    'description': item['snippet']['description']
                }
                videos.append(video)
            return videos
        except Exception as e:
            logger.error(f"Error fetching YouTube search results: {str(e)}")
            return []
