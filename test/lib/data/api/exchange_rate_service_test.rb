require 'test_helper'

class ExchangeRateServiceTest < ActiveSupport::TestCase
  setup do
    @prev_date = (Date.today - 1).to_s
  end

  test 'mock service and save' do
    stub_request(:get, "https://openexchangerates.org/api/historical/#{@prev_date}.json?app_id=test&base=USD").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'openexchangerates.org', 'User-Agent'=>'Ruby'}).
        to_return(status: 500).times(1).then.to_return(:status => 200, :body => '{"timestamp": 1481871613,"base": "USD","rates": {"YER": 250.109299,"ZAR": 13.94511,"ZMK": 5253.075255}}', :headers => {})

    TimeHelper.expects(:exponential_drop_off).returns(0.001)

    Data::Api::ExchangeRateService.call_service
    assert_equal(ExchangeRate.all.count, 1)
  end

  test 'mock service and fail' do
    stub_request(:get, "https://openexchangerates.org/api/historical/#{@prev_date}.json?app_id=test&base=USD").
        with(:headers => {'Accept'=>'*/*', 'Accept-Encoding'=>'gzip;q=1.0,deflate;q=0.6,identity;q=0.3', 'Host'=>'openexchangerates.org', 'User-Agent'=>'Ruby'}).
        to_return(status: 500).times(3)

    TimeHelper.expects(:exponential_drop_off).times(3).returns(0.001)
    ActionMailer::Base.deliveries.clear

    Data::Api::ExchangeRateService.call_service
    assert_equal(ExchangeRate.all.count, 0)
    assert_equal(1, ActionMailer::Base.deliveries.count)
  end

end