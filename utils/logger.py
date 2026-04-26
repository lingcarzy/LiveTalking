import logging
 
# 配置日志器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
if not logger.handlers:
	# Force UTF-8 to avoid Chinese mojibake across different Windows code pages.
	fhandler = logging.FileHandler('livetalking.log', encoding='utf-8')
	fhandler.setFormatter(formatter)
	fhandler.setLevel(logging.INFO)
	logger.addHandler(fhandler)

# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# sformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(sformatter)
# logger.addHandler(handler)