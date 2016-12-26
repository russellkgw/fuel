require 'test_helper'

class OilPriceServiceTest < ActiveSupport::TestCase
  setup do
    @prev_date = '2016-12-19'
  end

  test 'mock service and save' do
    stub_request(:get, "https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=test&end_date=#{@prev_date}&start_date=#{@prev_date}").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'www.quandl.com', 'User-Agent'=>'Ruby'}).to_return(status: 500).times(1).then.
        to_return(:status => 200, :body => '{"dataset":{"id":11835629,"dataset_code":"PET_RBRTE_D","database_code":"EIA","name":"Europe Brent Spot Price FOB, Daily","description":"Units = Dollars per Barrel. Europe Brent Spot Price FOB","refreshed_at":"2016-12-26T13:42:46.305Z","newest_available_date":"2016-12-19","oldest_available_date":"1987-05-20","column_names":["Date","Value"],"frequency":"daily","type":"Time Series","premium":false,"limit":null,"transform":null,"column_index":null,"start_date":"2016-12-19","end_date":"2016-12-19","data":[["2016-12-19",53.53]],"collapse":null,"order":null,"database_id":661}}', :headers => {})

    TimeHelper.expects(:exponential_drop_off).returns(0.001)
    Date.expects(:today).times(2).returns(Date.parse("2016-12-20"))

    Data::Api::OilPriceService.call_service
    assert_equal(OilPrice.all.count, 1)
  end

  test 'mock service, dont save when no data' do
    stub_request(:get, "https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=test&end_date=#{@prev_date}&start_date=#{@prev_date}").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'www.quandl.com', 'User-Agent'=>'Ruby'}).
        to_return(:status => 200, :body => '{"dataset":{"id":11835629,"dataset_code":"PET_RBRTE_D","database_code":"EIA","name":"Europe Brent Spot Price FOB, Daily","description":"Units = Dollars per Barrel. Europe Brent Spot Price FOB","refreshed_at":"2016-12-26T13:42:46.305Z","newest_available_date":"2016-12-19","oldest_available_date":"1987-05-20","column_names":["Date","Value"],"frequency":"daily","type":"Time Series","premium":false,"limit":null,"transform":null,"column_index":null,"start_date":"2016-12-19","end_date":"2016-12-19","data":[],"collapse":null,"order":null,"database_id":661}}', :headers => {})

    Date.expects(:today).times(1).returns(Date.parse("2016-12-20"))

    Data::Api::OilPriceService.call_service
    assert_equal(OilPrice.all.count, 0)
  end

  test 'mock service, dont save when no is weekend' do
    stub_request(:get, "https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=test&end_date=#{@prev_date}&start_date=#{@prev_date}").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'www.quandl.com', 'User-Agent'=>'Ruby'}).
        to_return(:status => 200, :body => '{"dataset":{"id":11835629,"dataset_code":"PET_RBRTE_D","database_code":"EIA","name":"Europe Brent Spot Price FOB, Daily","description":"Units = Dollars per Barrel. Europe Brent Spot Price FOB","refreshed_at":"2016-12-26T13:42:46.305Z","newest_available_date":"2016-12-19","oldest_available_date":"1987-05-20","column_names":["Date","Value"],"frequency":"daily","type":"Time Series","premium":false,"limit":null,"transform":null,"column_index":null,"start_date":"2016-12-19","end_date":"2016-12-19","data":[],"collapse":null,"order":null,"database_id":661}}', :headers => {})

    Date.expects(:today).times(1).returns(Date.parse("2016-12-18"))

    Data::Api::OilPriceService.call_service
    assert_equal(OilPrice.all.count, 0)
  end

  test 'mock service, dont save when data already exists' do
    stub_request(:get, "https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=test&end_date=#{@prev_date}&start_date=#{@prev_date}").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'www.quandl.com', 'User-Agent'=>'Ruby'}).
        to_return(:status => 200, :body => '{"dataset":{"id":11835629,"dataset_code":"PET_RBRTE_D","database_code":"EIA","name":"Europe Brent Spot Price FOB, Daily","description":"Units = Dollars per Barrel. Europe Brent Spot Price FOB","refreshed_at":"2016-12-26T13:42:46.305Z","newest_available_date":"2016-12-19","oldest_available_date":"1987-05-20","column_names":["Date","Value"],"frequency":"daily","type":"Time Series","premium":false,"limit":null,"transform":null,"column_index":null,"start_date":"2016-12-19","end_date":"2016-12-19","data":[],"collapse":null,"order":null,"database_id":661}}', :headers => {})

    Date.expects(:today).times(1).returns(Date.parse("2016-12-20"))

    OilPrice.create(date: "2016-12-19")

    Data::Api::OilPriceService.call_service
    assert_equal(OilPrice.all.count, 1)
  end

  test 'mock service and fail' do
    stub_request(:get, "https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=test&end_date=#{@prev_date}&start_date=#{@prev_date}").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'www.quandl.com', 'User-Agent'=>'Ruby'}).to_return(status: 500).times(3)

    TimeHelper.expects(:exponential_drop_off).times(3).returns(0.001)
    Date.expects(:today).times(3).returns(Date.parse("2016-12-20"))
    ActionMailer::Base.deliveries.clear

    Data::Api::OilPriceService.call_service
    assert_equal(OilPrice.all.count, 0)
    assert_equal(1, ActionMailer::Base.deliveries.count)
  end

end