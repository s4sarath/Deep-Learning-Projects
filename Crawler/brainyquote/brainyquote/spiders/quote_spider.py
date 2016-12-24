
from scrapy import Spider
from scrapy.selector import Selector
from scrapy import Request
import re

import pandas as pd
from brainyquote.items import BrainyquoteItem

class QuoteSpider(Spider):
    name = "brainy_quote"
    allowed_domains = ["brainyquote.com"]
    start_urls = [
        "https://www.brainyquote.com/quotes/topics.html",
    ]




    def parse_results(self, response ):

        item = BrainyquoteItem()
        sel = Selector(response)
        quotes_list = sel.xpath('//div[@id="quotesList"]/div/div/span/a/text()')
        authors_list = sel.xpath('//*[@id="quotesList"]/div/div/div[@class="bq-aut"]/a/text()')
        keywords_list_ = sel.xpath('//*[@id="quotesList"]/div/div/div[@class="body bq_boxyRelatedLeft bqBlackLink"]/a')
        onclicks = keywords_list_.xpath('@onclick').extract()
        keywords = keywords_list_.xpath('text()').extract()

        topic = response.url.split('/')[-1].split('.html')[0].split('_')[-1]
        topic_name = re.sub(ur'[0-9]', '', topic).strip()
        keywords_store = []

        print 'Keywords' , len(keywords)
        print 'Onclickss', len(onclicks)
        count = 0
        for i in xrange(len(keywords)):
            checker = onclicks[i].split(',')[-2]
            if i == 0:
                temp = []
                temp.append(keywords[i].lower())
                continue
            if checker == u"'0'":
                print count
                count = count + 1
                keywords_store.append(temp)
                temp = []
                temp.append(keywords[i].lower())
                continue
            temp.append(keywords[i].lower())
            if i == len(keywords) - 1:
                keywords_store.append(temp)


        authors = authors_list.extract()
        quotes = quotes_list.extract()



        print len(authors)
        print len(quotes)
        print len(keywords_store)
        
        data = []
        for i in xrange(len(quotes)):

            print authors[i] , keywords_store[i]
            data.append([authors[i] , quotes[i] , keywords_store[i] , topic_name])

        df = pd.DataFrame(data)
        df.to_csv('{}.csv'.format(topic), encoding = 'utf-8')
        yield item

        

        


    def parse_url(self, response):

        print "Inside"
        page_list = Selector(response).xpath('//ul[@class="pagination bqNPgn pagination-sm "]/li/a/text()')
        last_page = page_list.extract()[-2].strip()
        return last_page
        # page_url = response.url
        # print "Page url" , page_url
        # yield Request('https://www.brainyquote.com/quotes/topics/topic_age2.html', self.parse_results)
        # # for i in xrange(int(last_page)):
        # #     if i==0:
        # #         yield Request(page_url, self.parse_results)
        # #     else:
        # #         yield Request(page_url.split('.html')[0]+str(i+1)+'.html', self.parse_results)
        # #     break

    def parse(self, response, last_page = 39):



        topics_list = Selector(response).xpath('//tr[@valign="top"]/td/div/div')

        print "Topics List" , len(topics_list) 
        for topics in topics_list:
            url = topics.xpath('a/@href').extract()[0]
            topic_name = topics.xpath('a/text()').extract()[0]
            topic_url = 'http://www.brainyquote.com'+ url
            
            item = BrainyquoteItem()
            for i in xrange(last_page):
                if i == 0:
                    item['topics'] = topic_name
                    data = Request(topic_url, self.parse_results)
                    yield data
                else:
                    data = Request(topic_url.split('.html')[0]+str(i+1)+'.html', self.parse_results)
                    yield data
                
            
            


        # for topics in topics_list:
        #     item = BrainyquoteItem()

        #     question.xpath(
        #         'a[@class="question-hyperlink"]/text()').extract()[0]

        #     item['url'] = question 
        #     print item
        #     yield item
            # item = StackItem()
            # item['title'] = question.xpath(
            #     'a[@class="question-hyperlink"]/text()').extract()[0]
            # item['url'] = question.xpath(
            #     'a[@class="question-hyperlink"]/@href').extract()[0]
            # yield item


 
# /html/body/div[6]/div[3]/div[1]/div/table/tbody/tr/td

# /html/body/div[6]/div[3]/div[1]/div/table/tbody/tr/td[1]

# /html/body/div[6]/div[3]/div[1]/div/table/tbody/tr/td[1]/div[1]

# /html/body/div[6]/div[3]/div[1]/div/table/tbody/tr/td[1]/div[1]/div

# /html/body/div[6]/div[3]/div[1]/div/table/tbody/tr/td[1]/div[1]/div/a

