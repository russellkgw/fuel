require 'net/http'

class Data::Api::BtcPriceService
  class Error < RuntimeError; end
  class IntegrationError < Error; end

  include TimeHelper
  include ErrorHelper

  def self.call_service
    new.call_service
  end

  def call_service()
    call_dev()
  end

  def call_dev()
    prev_date = Date.today - 1
    start_date = BtcPrice.order(date: :desc).first.date + 1

    return if start_date > prev_date

    puts ("call btc price for: #{start_date.to_s} - #{prev_date.to_s}")

    uri = URI("https://www.quandl.com/api/v3/datasets/BITFINEX/BTCUSD.json?api_key=#{Rails.application.secrets.oil_price_key}&start_date=#{start_date.to_s}&end_date=#{prev_date.to_s}")
    service_request = Net::HTTP.get_response(uri)

    # byebug

    if service_request.response.is_a?(Net::HTTPSuccess)
      data = JSON.parse(service_request.body).to_h['dataset']['data']
      save_btc_price(data) if data.any?
      puts ('SUCCESS!')
    else
      raise IntegrationError.new("Unable to successfully call btc price api, error: #{service_request.body.to_s}")
    end
  end

  def save_btc_price(data)
    data.each do |item|
      next if BtcPrice.find_by_date(item[0]).present?

      BtcPrice.create(date: item[0],
                      high: item[1],
                      low: item[2],
                      mid: item[3],
                      last: item[4],
                      bid: item[5],
                      ask: item[6],
                      volume: item[7])
    end
  end
end