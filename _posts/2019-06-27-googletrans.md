---
layout:     post
title:      Googletrans
subtitle:   
date:       2019-06-27
author:     JD
header-img: img/post-jd-googletrans.jpg
catalog: true
tags:
    - googletrans
---

谷歌翻译库API，[官网在此](https://py-googletrans.readthedocs.io/en/latest/)

安装`googletrans`

    pip install googletrans

引用`googletrans`

    from googletrans import Translator

设置`service_url`可以选择不同翻译服务的url。`service_url`是列表，那程序会随机选取一个来使用。

    googletrans.Translator(service_urls=None, user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)'

参数详情如下

- service_urls (a sequence of strings) – google translate url list. URLs will be used randomly. For example ['translate.google.com', 'translate.google.co.kr']
- user_agent (str) – the User-Agent header to send when making requests.
- proxies (dictionary) – proxies configuration. Dictionary mapping protocol or protocol and host to the URL of the proxy For example {'http': 'foo.bar:3128', 'http://host.name': 'foo.bar:4012'}
- timeout (number or a double of numbers) – Definition of timeout for Requests library. Will be used by every request.

将文本(可以为列表)从一种语言翻译成另外一种语言。

    translate(text, dest='en', src='auto')

参数详情如下

- text (UTF-8 str; unicode; string sequence (list, tuple, iterator, generator)) – The source text(s) to be translated. Batch translation is supported via sequence input.
- dest – The language to translate the source text into. The value should be one of the language codes listed in googletrans.LANGUAGES or one of the language names listed in googletrans.LANGCODES.
- dest – str; unicode
- src – The language of the source text. The value should be one of the language codes listed in googletrans.LANGUAGES or one of the language names listed in googletrans.LANGCODES. If a language is not specified, the system will attempt to identify the source language automatically.
- src – str; unicode

`translate`返回`Translated`，`translate(text, dest='en', src='auto').text`得到翻译结果对象。

- src – source langauge (default: auto)
- dest – destination language (default: en)
- origin – original text
- text – translated text
- pronunciation – pronunciation

下面有具体案例（将任何语言转成英文）

    from googletrans import Translator
    
    def trans(source):
        translator = Translator(service_urls=['translate.google.cn'])
        dest, src = 'auto', 'en'
        trans_result = translator.translate(source, dest=dest, src=src)
        return trans_result.text

