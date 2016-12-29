require 'test_helper'

class OilPriceServiceTest < ActiveSupport::TestCase
  setup do
    @prev_date = '2016-12-28'
    @prev_month = '2016-11-28'
    @body_response = '{"dataset":{"id":11835629,"dataset_code":"PET_RBRTE_D","database_code":"EIA","name":"Europe Brent Spot Price FOB, Daily","description":"Units = Dollars per Barrel. Europe Brent Spot Price FOB","refreshed_at":"2016-12-29T13:43:17.748Z","newest_available_date":"2016-12-19","oldest_available_date":"1987-05-20","column_names":["Date","Value"],"frequency":"daily","type":"Time Series","premium":false,"limit":null,"transform":null,"column_index":null,"start_date":"2016-11-28","end_date":"2016-12-19","data":[["2016-12-19",53.53],["2016-12-16",54.15],["2016-12-15",51.72],["2016-12-14",53.15],["2016-12-13",53.28],["2016-12-12",53.99],["2016-12-09",52.19],["2016-12-08",51.6],["2016-12-07",51.9],["2016-12-06",52.31],["2016-12-05",53.3],["2016-12-02",52.35],["2016-12-01",52.28],["2016-11-30",47.95],["2016-11-29",44.68],["2016-11-28",46.64]],"collapse":null,"order":null,"database_id":661}}'
  end

  test 'mock service and save' do
    stub_request(:get, "https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=test&end_date=#{@prev_date}&start_date=#{@prev_month}").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'www.quandl.com', 'User-Agent'=>'Ruby'}).to_return(status: 500).times(1).then.
        to_return(:status => 200, :body => @body_response, :headers => {})

    TimeHelper.expects(:exponential_drop_off).returns(0.001)
    Date.expects(:today).times(2).returns(Date.parse("2016-12-29"))

    Data::Api::OilPriceService.call_service
    assert_equal(OilPrice.all.count, 16)
  end

  test 'mock service, dont save when no data' do
    stub_request(:get, "https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=test&end_date=#{@prev_date}&start_date=#{@prev_month}").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'www.quandl.com', 'User-Agent'=>'Ruby'}).
        to_return(:status => 200, :body => '{"dataset":{"id":11835629,"dataset_code":"PET_RBRTE_D","database_code":"EIA","name":"Europe Brent Spot Price FOB, Daily","description":"Units = Dollars per Barrel. Europe Brent Spot Price FOB","refreshed_at":"2016-12-26T13:42:46.305Z","newest_available_date":"2016-12-19","oldest_available_date":"1987-05-20","column_names":["Date","Value"],"frequency":"daily","type":"Time Series","premium":false,"limit":null,"transform":null,"column_index":null,"start_date":"2016-12-19","end_date":"2016-12-19","data":[],"collapse":null,"order":null,"database_id":661}}', :headers => {})

    Date.expects(:today).times(1).returns(Date.parse("2016-12-29"))

    Data::Api::OilPriceService.call_service
    assert_equal(OilPrice.all.count, 0)
  end

  test 'mock service, dont save when no is weekend' do
    stub_request(:get, "https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=test&end_date=#{@prev_date}&start_date=#{@prev_month}").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'www.quandl.com', 'User-Agent'=>'Ruby'}).
        to_return(:status => 200, :body => '{"dataset":{"id":11835629,"dataset_code":"PET_RBRTE_D","database_code":"EIA","name":"Europe Brent Spot Price FOB, Daily","description":"Units = Dollars per Barrel. Europe Brent Spot Price FOB","refreshed_at":"2016-12-26T13:42:46.305Z","newest_available_date":"2016-12-19","oldest_available_date":"1987-05-20","column_names":["Date","Value"],"frequency":"daily","type":"Time Series","premium":false,"limit":null,"transform":null,"column_index":null,"start_date":"2016-12-19","end_date":"2016-12-19","data":[],"collapse":null,"order":null,"database_id":661}}', :headers => {})

    Date.expects(:today).times(1).returns(Date.parse("2016-12-25"))

    Data::Api::OilPriceService.call_service
    assert_equal(OilPrice.all.count, 0)
  end

  test 'mock service, dont save when data already exists' do
    stub_request(:get, "https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=test&end_date=#{@prev_date}&start_date=#{@prev_month}").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'www.quandl.com', 'User-Agent'=>'Ruby'}).
        to_return(:status => 200, :body => @body_response, :headers => {})

    Date.expects(:today).times(1).returns(Date.parse("2016-12-29"))

    (Date.new(2016, 11, 28)..Date.new(2016, 12, 28)).each { |date| OilPrice.create(date: date) }
    pre_call_count = OilPrice.all.count

    Data::Api::OilPriceService.call_service
    assert_equal(pre_call_count, 31)
  end

  test 'mock service, dont save or update existing.' do
    stub_request(:get, "https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=test&end_date=#{@prev_date}&start_date=#{@prev_month}").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'www.quandl.com', 'User-Agent'=>'Ruby'}).
        to_return(:status => 200, :body => @body_response, :headers => {})

    Date.expects(:today).times(1).returns(Date.parse("2016-12-29"))

    (Date.new(2016, 11, 28)..Date.new(2016, 12, 11)).each { |date| OilPrice.create(date: date, price: 1.0) }
    count_1 = OilPrice.where(price: 1.0).count

    Data::Api::OilPriceService.call_service
    count_n1 = OilPrice.where('price != 1.0').count

    assert_equal(count_1, 14)
    assert_equal(count_n1, 6)
  end

  test 'mock service and fail' do
    stub_request(:get, "https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.json?api_key=test&end_date=#{@prev_date}&start_date=#{@prev_month}").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'www.quandl.com', 'User-Agent'=>'Ruby'}).to_return(status: 500).times(3)

    TimeHelper.expects(:exponential_drop_off).times(3).returns(0.001)
    Date.expects(:today).times(3).returns(Date.parse("2016-12-29"))
    ActionMailer::Base.deliveries.clear

    Data::Api::OilPriceService.call_service
    assert_equal(OilPrice.all.count, 0)
    assert_equal(1, ActionMailer::Base.deliveries.count)
  end
end