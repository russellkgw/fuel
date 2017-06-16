class Data::Api::ExchangeRateFutureService
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
    start_date = ExchangeFuture.order(date: :desc).first.date + 1

    return if start_date > prev_date

    puts ("call exchange future for: #{start_date.to_s} - #{prev_date.to_s}")

    uri = URI("https://www.quandl.com/api/v3/datasets/CHRIS/ICE_ZR1.json?api_key=#{Rails.application.secrets.oil_price_key}&start_date=#{start_date.to_s}&end_date=#{prev_date.to_s}")
    service_request = Net::HTTP.get_response(uri)

    if service_request.response.is_a?(Net::HTTPSuccess)
      data = JSON.parse(service_request.body).to_h['dataset']['data']
      save_exchange_future(data) if data.any?
      puts ('SUCCESS!')
    else
      raise IntegrationError.new("Unable to successfully call exchange future api, error: #{service_request.body.to_s}")
    end
  end

  def save_exchange_future(data)
    data.each do |item|
      next if ExchangeFuture.find_by_date(item[0]).present?

      ExchangeFuture.create(base: 'USD',
                            currency: 'ZAR',
                            date: item[0],
                            settle: item[4],
                            wave: item[6],
                            volume: item[7],
                            efp_volume: item[9],
                            efs_volume: item[10],
                            block_volume: item[11],
                            source: 'https://www.quandl.com/data/CHRIS/ICE_ZR1-US-Dollar-South-African-Rand-Futures-Continuous-Contract-1-ZR1-Front-Month')
    end
  end
end